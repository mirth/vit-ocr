import torch
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import string

chars = list(string.ascii_lowercase + string.digits)
char_by_id = {i:c for i, c in enumerate(chars)}
id_by_char = {c:i for i, c in enumerate(chars)}
def encode(s):
    return [id_by_char[c] for c in s]
def decode(ids):
    return ''.join([char_by_id[id] for id in ids])


class MyDataset(Dataset):
    def __init__(self, dataset_rootdir, df, max_length):
        self.data = df.values
        self.dataset_rootdir = dataset_rootdir
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        y, x = self.data[idx]

        x = Image.open(f'{self.dataset_rootdir}/' + x)
        x = self.transform(x)

        y = torch.from_numpy(np.array(encode(y)))
        # y = self.tokenizer(
        #     y,
        #     return_tensors="pt",
        #     max_length=self.max_length,
        #     padding="max_length",
        #     truncation=True,
        # )
        # y = y["input_ids"]

        return x, y
    
def load_df(dataset_rootdir, nrows=None):
    from sklearn.model_selection import train_test_split
    import pandas as pd

    df = pd.read_csv(f"{dataset_rootdir}/data.csv", nrows=nrows)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    return df_train, df_test

if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv("dataset0/data.csv", nrows=100)

    dataset = MyDataset(df)
    x, y = dataset[0]
    print(y)
    print(x)
    print(x.sum())
