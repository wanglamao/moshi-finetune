import logging
from typing import Iterator

import torch
import torch.cuda
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from finetune.args import TrainArgs
from moshi.utils.compile import no_compile

from .data.data_loader import Batch
from .distributed import get_rank, get_world_size
from .loss import compute_loss_with_mask
from .utils import TrainState

logger = logging.getLogger("eval")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def evaluate(
    model: FullyShardedDataParallel,
    eval_data_loader: Iterator[Batch],
    state: TrainState,
    args: TrainArgs,
    max_eval_batches: int = 100,  # Maximum batches per GPU for eval
):
    num_batches = torch.tensor([0], device="cuda", dtype=torch.long)
    num_samples = torch.tensor([0], device="cuda", dtype=torch.long)

    text_loss = torch.tensor(0.0).cuda()
    audio_loss = torch.tensor(0.0).cuda()
    model.eval()

    # Disable torch.compile during evaluation to avoid gradient tracking issues
    with no_compile():
        eval_iter = iter(eval_data_loader)
        # Each GPU processes up to max_eval_batches batches
        max_batches_per_gpu = max_eval_batches

        for i in range(max_batches_per_gpu):
            # Try to get next batch
            try:
                batch = next(eval_iter)
                has_data = True
            except StopIteration:
                has_data = False

            # Synchronize across all ranks - if any rank runs out of data, all should stop
            has_data_tensor = torch.tensor(has_data, dtype=torch.bool, device="cuda")
            if get_world_size() > 1:
                torch.distributed.all_reduce(has_data_tensor, op=torch.distributed.ReduceOp.MIN)

            if not has_data_tensor.item():
                # At least one rank ran out of data, all ranks should stop
                break

            num_batches += 1
            batch_size = batch.codes.shape[0]
            num_samples += batch_size

            with torch.no_grad():
                codes = batch.codes
                condition_tensors = None
                if batch.condition_attributes is not None:
                    condition_tensors = model.condition_provider.prepare(
                        batch.condition_attributes
                    )

                output = model(codes=codes, condition_tensors=condition_tensors)
                text_loss += compute_loss_with_mask(
                    output.text_logits,
                    codes[:, : model.audio_offset],
                    output.text_mask,
                    mode="text",
                    text_padding_weight=args.text_padding_weight,
                    text_padding_ids={
                        model.text_padding_token_id,
                        model.end_of_text_padding_id,
                    },
                )
                # STT models have dep_q=0, so skip audio loss
                if model.dep_q > 0:
                    audio_loss += compute_loss_with_mask(
                        output.logits,
                        codes[:, model.audio_offset : model.audio_offset + model.dep_q],
                        output.mask,
                        mode="audio",
                        first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
                    )
    eval_loss = text_loss + audio_loss

    # Gather batch counts from all ranks
    all_num_batches = [torch.zeros_like(num_batches) for _ in range(get_world_size())]
    all_num_samples = [torch.zeros_like(num_samples) for _ in range(get_world_size())]

    torch.distributed.all_gather(all_num_batches, num_batches)
    torch.distributed.all_gather(all_num_samples, num_samples)

    total_num_batches = int(torch.tensor(all_num_batches).sum().item())
    total_num_samples = int(torch.tensor(all_num_samples).sum().item())

    main_logger_info(f"Eval finished! Total batches: {total_num_batches}, Total samples: {total_num_samples}")

    if total_num_batches == 0:
        main_logger_info("Warning: No eval batches processed, skipping eval metrics update")
        return

    dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(text_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(audio_loss, op=dist.ReduceOp.SUM)
    text_loss /= total_num_batches
    audio_loss /= total_num_batches
    eval_loss /= total_num_batches

    state.this_eval_loss = eval_loss.item()
    state.this_eval_perplexity = (2**eval_loss).item()
    state.this_audio_loss = audio_loss.item()
    state.this_text_loss = text_loss.item()

    # train mode!
    model.train()
