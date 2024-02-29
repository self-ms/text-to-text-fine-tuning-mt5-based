from transformers import MT5Tokenizer
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import torch
import nlp


class Dtataframe2Dataset:

    def __init__(self,data_args):

        self.input_length = 128
        self.output_length = 128
        self.tokenizer = MT5Tokenizer.from_pretrained(data_args.dataset_tokenizer)
        df = pd.read_csv(data_args.dataset_file_name).dropna()[-20:]
        self.train_dataset, self.valid_dataset, self.test_dataset = self.__data_frame_spliter(data_frame=df)

        torch.save(self.train_dataset, data_args.train_file_path)
        torch.save(self.valid_dataset, data_args.valid_file_path)
        torch.save(self.test_dataset, data_args.test_file_path)


    def __data_frame_spliter(self, data_frame):
        train_size = int(len(data_frame)*0.6)
        val_size = int(len(data_frame)*0.2)
        test_size = int(len(data_frame)*0.1)

        train_data = data_frame.iloc[:train_size]
        valid_data = data_frame.iloc[train_size:train_size+val_size]
        test_data = data_frame.iloc[train_size+val_size:]

        train_dataset = nlp.Dataset.from_pandas(train_data)
        valid_dataset = nlp.Dataset.from_pandas(valid_data)
        test_dataset = nlp.Dataset.from_pandas(test_data)

        train_dataset = train_dataset.map(self.__convert_to_features, batched=True)
        valid_dataset = valid_dataset.map(self.__convert_to_features, batched=True, load_from_cache_file=False)
        test_dataset = test_dataset.map(self.__convert_to_features, batched=True, load_from_cache_file=False)

        columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
        train_dataset.set_format(type='torch', columns=columns)
        valid_dataset.set_format(type='torch', columns=columns)
        test_dataset.set_format(type='torch', columns=columns)

        return train_dataset, valid_dataset, test_dataset

    def __convert_to_features(self, example_batch):
        input_encodings = self.tokenizer.batch_encode_plus(example_batch['input_text'], pad_to_max_length=True, max_length=self.input_length)
        target_encodings = self.tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True, max_length=self.output_length)
        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'target_ids': target_encodings['input_ids'],
            'target_attention_mask': target_encodings['attention_mask']
        }
        return encodings




@dataclass
class T2TDataCollator: #(DataCollator)
    def __call__(self, batch: List) -> Dict[str, torch.Tensor]: #collate_batch
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example['input_ids'] for example in batch])
        lm_labels = torch.stack([example['target_ids'] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])


        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': lm_labels,
            'decoder_attention_mask': decoder_attention_mask
        }