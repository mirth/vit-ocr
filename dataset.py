import torch
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import tokenizer

class MyDataset(Dataset):
    def __init__(self, dataset_rootdir, df, max_length):
        self.data = df.values
        self.dataset_rootdir = dataset_rootdir
        self.max_length = max_length

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

        y = torch.from_numpy(np.array(tokenizer.encode(y)))

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
