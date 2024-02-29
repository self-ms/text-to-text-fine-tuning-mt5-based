from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    tokenizer_name: Optional[str] = field(default=None, metadata={
        "help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(default=None, metadata={
        "help": "Where do you want to store the pretrained models downloaded from s3"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_file_name: Optional[str] = field(default='Dataset/dataset.csv', metadata={"help": "Path for cached train dataset"}, )
    train_file_path: Optional[str] = field(default='Dataset/train_data.pt', metadata={"help": "Path for cached train dataset"}, )
    valid_file_path: Optional[str] = field(default='Dataset/valid_data.pt', metadata={"help": "Path for cached valid dataset"}, )
    test_file_path: Optional[str] = field(default='Dataset/test_data.pt', metadata={"help": "Path for cached valid dataset"}, )
    dataset_tokenizer: Optional[str] = field(default='google/mt5-base', metadata={"help": "Path for cached valid dataset"}, )
    max_len: Optional[int] = field(default=128, metadata={"help": "Max input length for the source text"}, )
    target_max_len: Optional[int] = field(default=128, metadata={"help": "Max input length for the target text"}, )