"""
WebDataset-based data loader for handling billions of audio samples efficiently.

WebDataset avoids filesystem bottlenecks by packaging data into tar archives,
which is critical when dealing with hundreds of millions of small audio files.
"""

import io
import json
import logging
import math
import traceback
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.distributed as dist
import webdataset as wds
from scipy.io import wavfile

from finetune.distributed import get_rank

from .interleaver import InterleavedTokenizer, Sample, dicho, Batch

logger = logging.getLogger("webdataset_loader")


def main_logger_info(message: str) -> None:
    if dist.is_initialized() and get_rank() == 0:
        logger.info(message)


def decode_audio(data: bytes, sample_rate: int) -> np.ndarray:
    """Decode audio bytes to numpy array."""
    # Read wav from bytes
    bio = io.BytesIO(data)
    sr, audio = wavfile.read(bio)

    # Resample if needed (basic implementation)
    if sr != sample_rate:
        logger.warning(
            f"Audio sample rate {sr} != target {sample_rate}, resampling not implemented"
        )

    # Convert to float32 and normalize
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)

    # Ensure mono (if stereo, average channels)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=-1)

    return audio


def create_webdataset_iterator(
    urls: str | list[str],
    instruct_tokenizer: InterleavedTokenizer,
    rank: int,
    world_size: int,
    shuffle: bool = False,
    shuffle_buffer: int = 1000,
    seed: int | None = None,
) -> Iterator[Sample]:
    """
    Create a WebDataset iterator for streaming audio data.

    Args:
        urls: Path pattern to webdataset shards (e.g., "data/stt_zh_webdataset/shard-{000000..000099}.tar")
              or list of tar file paths
        instruct_tokenizer: Tokenizer for processing audio
        rank: DDP rank
        world_size: DDP world size
        shuffle: Whether to shuffle samples
        shuffle_buffer: Size of shuffle buffer
        seed: Random seed for shuffling

    Yields:
        Sample objects ready for training
    """
    import sys

    # print(f"[DEBUG] create_webdataset_iterator STARTING", file=sys.stderr, flush=True)

    # Create a WebDatasetTokenizer wrapper that handles alignments from webdataset
    try:
        web_tokenizer = WebDatasetTokenizer(
            mimi=instruct_tokenizer.mimi,
            interleaver=instruct_tokenizer.interleaver,
            duration_sec=instruct_tokenizer.duration_sec,
            downmix_to_mono=instruct_tokenizer.downmix_to_mono,
        )
        # print("[DEBUG] WebDatasetTokenizer created successfully", flush=True)
    except Exception as e:
        # print(
        #     f"[DEBUG] Failed to create WebDatasetTokenizer: {e}\n{traceback.format_exc()}",
        #     flush=True,
        # )
        raise

    # Create dataset
    # print(
    #     f"[DEBUG] Creating wds.WebDataset with {len(urls) if isinstance(urls, list) else 'pattern'} shards",
    #     flush=True,
    # )
    # Note: workersplitter is already set by default in WebDataset to split_by_worker
    # nodesplitter handles DDP splitting across nodes
    dataset = wds.WebDataset(
        urls,
        nodesplitter=wds.split_by_node if world_size > 1 else None,
        shardshuffle=shuffle,
    )

    # Shuffle if requested
    if shuffle:
        import random

        rng = random.Random(seed)
        dataset = dataset.shuffle(shuffle_buffer, rng=rng)

    # Decode samples
    def decode_sample(sample):
        """Decode webdataset sample."""
        try:
            # Parse metadata - handle both bytes and dict (webdataset auto-decodes json)
            json_data = sample["json"]
            if isinstance(json_data, bytes):
                metadata = json.loads(json_data.decode("utf-8"))
            else:
                metadata = json_data

            # Decode audio
            audio = decode_audio(sample["wav"], instruct_tokenizer.mimi.sample_rate)

            # Get key for sample identification
            key = sample["__key__"]

            # Extract alignments from metadata
            alignments = metadata.get("alignments", [])

            return {
                "audio": audio,
                "metadata": metadata,
                "key": key,
                "alignments": alignments,
            }
        except Exception as e:
            logger.error(
                f"Failed to decode sample {sample.get('__key__', 'unknown')}: {e}\n{traceback.format_exc()}"
            )
            return None

    dataset = dataset.map(decode_sample)

    # Filter out failed decodes
    dataset = dataset.select(lambda x: x is not None)

    # Chunk audio into training segments
    def chunk_audio_sample(sample):
        """Split audio into fixed-duration chunks."""
        audio = sample["audio"]
        duration_sec = instruct_tokenizer.duration_sec
        sample_rate = instruct_tokenizer.mimi.sample_rate

        chunk_samples = int(duration_sec * sample_rate)
        n_chunks = max(1, int(np.ceil(len(audio) / chunk_samples)))

        chunks = []
        for i in range(n_chunks):
            start_idx = i * chunk_samples
            end_idx = min((i + 1) * chunk_samples, len(audio))
            chunk = audio[start_idx:end_idx]

            # Pad last chunk if needed
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

            chunks.append(
                {
                    "audio_chunk": chunk,
                    "start_time_sec": i * duration_sec,
                    "key": f"{sample['key']}_chunk{i}",
                    "metadata": sample["metadata"],
                    "alignments": sample["alignments"],
                }
            )

        return chunks

    # Flatten chunks
    dataset = dataset.compose(
        lambda source: (chunk for sample in source for chunk in chunk_audio_sample(sample))
    )

    # Convert to Sample objects using WebDatasetTokenizer
    sample_count = 0
    error_count = 0
    for item in dataset:
        try:
            audio_chunk = item["audio_chunk"]
            # Add channel dimension if needed
            if audio_chunk.ndim == 1:
                audio_chunk = audio_chunk[np.newaxis, :]

            sample = web_tokenizer(
                wav=audio_chunk,
                start_sec=item["start_time_sec"],
                path=item["key"],
                alignments=item["alignments"],
            )
            sample_count += 1
            # if sample_count <= 3:
            # logger.info(
            #     f"Successfully processed sample {sample_count}: {item.get('key', 'unknown')}"
            # )
            yield sample

        except Exception as e:
            error_count += 1
            logger.error(
                f"Failed to process chunk {item.get('key', 'unknown')}: {e}\n{traceback.format_exc()}"
            )
            if error_count >= 10:
                logger.error("Too many errors, stopping iteration")
                raise
            continue

    logger.info(
        f"create_webdataset_iterator finished: {sample_count} samples, {error_count} errors"
    )


def _build_webdataset_loader_inner(
    dataset: Iterator[Sample],
    batch_size: int,
) -> Iterator[Batch]:
    """Inner generator that batches samples."""
    import sys

    # print(
    #     "[DEBUG] _build_webdataset_loader_inner STARTING", file=sys.stderr, flush=True
    # )
    sample_list = []
    for sample in dataset:
        # print(f"[DEBUG] Got sample from dataset", file=sys.stderr, flush=True)
        assert sample.codes.dim() == 3
        assert len(sample.codes) == 1
        sample_list.append(sample)

        if len(sample_list) == batch_size:
            yield Batch.collate(sample_list)
            sample_list = []
    # print(
    #     "[DEBUG] _build_webdataset_loader_inner FINISHED (no more samples)",
    #     file=sys.stderr,
    #     flush=True,
    # )


def build_webdataset_loader(
    data_path: str,
    instruct_tokenizer: InterleavedTokenizer,
    batch_size: int,
    rank: int,
    world_size: int,
    shuffle: bool = False,
    shuffle_buffer: int = 1000,
    seed: int | None = None,
    is_eval: bool = False,
) -> Iterator[Batch]:
    """
    Build a data loader from WebDataset shards.

    Args:
        data_path: Path to webdataset directory or shard pattern
                   Examples:
                   - "data/stt_zh_webdataset" (will auto-detect shards)
                   - "data/stt_zh_webdataset/shard-{000000..000099}.tar" (explicit pattern)
        instruct_tokenizer: Tokenizer for processing audio
        batch_size: Batch size
        rank: DDP rank
        world_size: DDP world size
        shuffle: Whether to shuffle samples
        shuffle_buffer: Size of shuffle buffer for shuffling
        seed: Random seed
        is_eval: If True, iterate once; if False, loop infinitely

    Returns:
        Iterator of Batches
    """
    # Determine shard URLs (this runs immediately, not lazily)
    data_path_obj = Path(data_path)
    logger.info(
        f"build_webdataset_loader: data_path={data_path}, is_dir={data_path_obj.is_dir()}"
    )

    if data_path_obj.is_dir():
        # Auto-detect tar files in directory
        tar_files = sorted(data_path_obj.glob("*.tar"))
        if not tar_files:
            raise ValueError(f"No .tar files found in {data_path}")
        urls = [str(f) for f in tar_files]
        logger.info(f"Found {len(urls)} shard files: {urls[:3]}...")
    elif "{" in data_path and "}" in data_path:
        # Brace expansion pattern (e.g., "shard-{000000..000099}.tar")
        urls = data_path
        logger.info(f"Using brace expansion pattern: {urls}")
    else:
        raise ValueError(
            f"Invalid data_path: {data_path}. Must be a directory or brace expansion pattern."
        )

    # Loop infinitely for training, once for eval
    epoch = 1
    while True:
        logger.info(f"Starting epoch {epoch}")

        # Create dataset iterator for this epoch
        dataset = create_webdataset_iterator(
            urls=urls,
            instruct_tokenizer=instruct_tokenizer,
            rank=rank,
            world_size=world_size,
            shuffle=shuffle,
            shuffle_buffer=shuffle_buffer,
            seed=seed + epoch if seed is not None else None,  # Different seed per epoch
        )

        # Yield batches
        yield from _build_webdataset_loader_inner(dataset, batch_size)

        if is_eval:
            break

        logger.info(f"Rank {rank} finished epoch {epoch}")
        epoch += 1


class WebDatasetTokenizer:
    """Tokenizer for webdataset format that receives alignments directly."""

    def __init__(self, mimi, interleaver, duration_sec: float, downmix_to_mono: bool):
        self.mimi = mimi
        self.interleaver = interleaver
        self.duration_sec = duration_sec
        self.num_audio_frames = math.ceil(duration_sec * mimi.frame_rate)
        self.downmix_to_mono = downmix_to_mono

    def __call__(
        self, wav: np.ndarray, start_sec: float, path: str, alignments: list
    ) -> Sample:
        """Process audio with alignments provided directly (not from file)."""
        # `sphn.dataset_jsonl` yields wav shaped (channels, samples). For STT, force mono.
        if self.downmix_to_mono and wav.ndim == 2 and wav.shape[0] > 1:
            wav = wav.mean(axis=0, keepdims=True)

        with torch.no_grad():
            audio_tensor = torch.Tensor(wav).cuda()
            audio_tokens = self.mimi.encode(audio_tensor[:, None])
            audio_tokens = audio_tokens[..., : self.num_audio_frames]
            this_num_audio_frames = audio_tokens.shape[-1]
            audio_tokens = torch.nn.functional.pad(
                audio_tokens[..., : self.num_audio_frames],
                (0, self.num_audio_frames - this_num_audio_frames),
                value=self.interleaver.zero_padding,
            )
            audio_tokens = audio_tokens.view(1, -1, self.num_audio_frames)

            # Use alignments provided directly from webdataset
            start_alignment = dicho(alignments, start_sec)
            end_alignment = dicho(alignments, start_sec + self.duration_sec)
            alignments = [
                (a[0], (a[1][0] - start_sec, a[1][1] - start_sec), a[2])
                for a in alignments[start_alignment:end_alignment]
            ]

            text_tokens = self.interleaver.prepare_item(
                alignments, this_num_audio_frames
            )
            text_tokens = torch.nn.functional.pad(
                text_tokens,
                (0, self.num_audio_frames - text_tokens.shape[-1]),
                value=self.interleaver.zero_padding,
            )

            codes = torch.cat([text_tokens, audio_tokens], dim=1)
            return Sample(codes, None)
