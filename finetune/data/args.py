import logging
from dataclasses import dataclass

from simple_parsing.helpers import Serializable

logger = logging.getLogger("data")


@dataclass()
class DataArgs(Serializable):
    """
     Arguments for data loading. Train and eval data should be jsonl files
    with  "path" and "duration" fields for each audio .wav file.

    For webdataset format, train_data should point to a directory containing
    .tar shards or a brace expansion pattern like "path/shard-{000000..000099}.tar"
    """

    train_data: str = ""
    shuffle: bool = False
    eval_data: str = ""

    # WebDataset options
    use_webdataset: bool = False
    webdataset_shuffle_buffer: int = 1000
