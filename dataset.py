import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, df):
        self.data = df.values
        self.max_length = 16

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

        x = Image.open('dataset0/' + x)
        x = self.transform(x)

        y = self.tokenizer(
            y,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        y = y["input_ids"]

        return x, y
    
def load_df(nrows=None):
    from sklearn.model_selection import train_test_split
    import pandas as pd

    df = pd.read_csv("dataset0/data.csv", nrows=nrows)
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
