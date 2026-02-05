import logging
from typing import Any, Iterator

from .args import DataArgs
from .dataset import build_dataset
from .interleaver import Batch

logger = logging.getLogger("dataloader")

def build_data_loader(
    instruct_tokenizer: Any,
    args: DataArgs,
    batch_size: int,
    seed: int | None,
    rank: int,
    world_size: int,
    is_eval: bool,
) -> Iterator[Batch]:
    if is_eval:
        assert args.eval_data != "", "No eval data provided."
    pretrain_data = args.train_data if not is_eval else args.eval_data

    logger.info(f"args if {args}")

    # Check if using webdataset format
    if args.use_webdataset:
        logger.info("Using WebDataset data loader.")
        from .webdataset_loader import build_webdataset_loader

        yield from build_webdataset_loader(
            data_path=pretrain_data,
            instruct_tokenizer=instruct_tokenizer,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            shuffle=not is_eval and args.shuffle,
            shuffle_buffer=args.webdataset_shuffle_buffer,
            seed=seed,
            is_eval=is_eval,
        )
        return
    else:
        # Original jsonl-based loader
        dataset = build_dataset(
            pretrain_data=pretrain_data,
            instruct_tokenizer=instruct_tokenizer,
            seed=seed,
            rank=rank,
            world_size=world_size,
            is_eval=is_eval,
            shuffle_pretrain=args.shuffle,
        )

        sample_list = []
        for sample in dataset:
            assert sample.codes.dim() == 3
            assert len(sample.codes) == 1
            sample_list.append(sample)

            if len(sample_list) == batch_size:
                yield Batch.collate(sample_list)
                sample_list = []
