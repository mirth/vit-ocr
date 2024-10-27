import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        # self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self):
        return 10#len(self.data)

    def __getitem__(self, idx):

        x = torch.ones(3, 224, 224)
        y = "hello eirugre"
        max_length = 32
        y = self.tokenizer(
            y,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        y = y["input_ids"]

        return x, y
    

if __name__ == '__main__':
    data = ["hello", "world"]
    dataset = MyDataset(data)
    x, y = dataset[0]
    print(y)
