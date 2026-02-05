"""
WebDataset-based data loader for handling billions of audio samples efficiently.

WebDataset avoids filesystem bottlenecks by packaging data into tar archives,
which is critical when dealing with hundreds of millions of small audio files.
"""

import io
import json
import logging
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
    main_logger_info(f"Creating WebDataset from: {urls}")

    # Create a WebDatasetTokenizer wrapper that handles alignments from webdataset
    web_tokenizer = WebDatasetTokenizer(
        mimi=instruct_tokenizer.mimi,
        interleaver=instruct_tokenizer.interleaver,
        duration_sec=instruct_tokenizer.duration_sec,
        downmix_to_mono=instruct_tokenizer.downmix_to_mono,
    )

    # Create dataset
    dataset = wds.WebDataset(urls, nodesplitter=wds.split_by_node, shardshuffle=shuffle)

    # Split by worker (for DDP)
    if world_size > 1:
        dataset = dataset.split_by_worker

    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer, rng=np.random.default_rng(seed))

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
            logger.warning(f"Failed to decode sample {sample.get('__key__', 'unknown')}: {e}")
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
            yield sample

        except Exception as e:
            logger.warning(f"Failed to process chunk {item.get('key', 'unknown')}: {e}")
            continue


def build_webdataset_loader(
    data_path: str,
    instruct_tokenizer: InterleavedTokenizer,
    batch_size: int,
    rank: int,
    world_size: int,
    shuffle: bool = False,
    shuffle_buffer: int = 1000,
    seed: int | None = None,
) -> Iterator:
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

    Yields:
        Batches of samples
    """

    # Determine shard URLs
    data_path_obj = Path(data_path)
    main_logger_info(f"data_path_obj.is_dir() {data_path_obj.is_dir()}")

    if data_path_obj.is_dir():

        # Auto-detect tar files in directory
        tar_files = sorted(data_path_obj.glob("*.tar"))
        if not tar_files:
            raise ValueError(f"No .tar files found in {data_path}")
        urls = [str(f) for f in tar_files]
        main_logger_info(f"Found {len(urls)} shard files")
    elif "{" in data_path and "}" in data_path:
        # Brace expansion pattern (e.g., "shard-{000000..000099}.tar")
        urls = data_path
    else:
        raise ValueError(
            f"Invalid data_path: {data_path}. Must be a directory or brace expansion pattern."
        )

    # Create dataset iterator
    dataset = create_webdataset_iterator(
        urls=urls,
        instruct_tokenizer=instruct_tokenizer,
        rank=rank,
        world_size=world_size,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )

    # Batch samples
    sample_list = []
    for sample in dataset:
        assert sample.codes.dim() == 3
        assert len(sample.codes) == 1
        sample_list.append(sample)

        if len(sample_list) == batch_size:
            yield Batch.collate(sample_list)
            sample_list = []


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
