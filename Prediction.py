import torch
import pandas as pd
import numpy as np

from transformers import MT5ForConditionalGeneration, MT5Tokenizer

if __name__ == "__main__":
    model = MT5ForConditionalGeneration.from_pretrained('output').to('cpu')
    tokenizer = MT5Tokenizer.from_pretrained('output')

    text ="Hi there, how can I help?"
    input_encodings = tokenizer.encode_plus(text, pad_to_max_length=True, max_length=512)

    outs = model.generate(input_ids=torch.tensor([input_encodings['input_ids']]), attention_mask=torch.tensor([input_encodings['attention_mask']]), max_length=128, early_stopping=True)
    outs = [tokenizer.decode(ids) for ids in outs]
    print(outs)

