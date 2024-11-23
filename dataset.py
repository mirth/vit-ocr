import torch
import cv2
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torchvision import transforms
import tokenizer
import augs


class MyDataset(Dataset):
    def __init__(self, dataset_rootdir, df, max_length):
        self.data = df[['image', 'text']].values
        self.dataset_rootdir = dataset_rootdir
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = cv2.imread(f'{self.dataset_rootdir}/' + x)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = augs.resize0(image=x)['image'].float()

        y = torch.from_numpy(np.array(tokenizer.encode(y, max_length=self.max_length)))

        return x, y
    
def load_df(dataset_rootdir, nrows=None):
    from sklearn.model_selection import train_test_split
    import pandas as pd

    df = pd.read_csv(f"{dataset_rootdir}/data.csv", nrows=nrows)
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)

    return df_train, df_test

if __name__ == '__main__':
    import pandas as pd
    from tqdm import tqdm

    # df = pd.read_csv("datasets/dataset8/data.csv", nrows=100)
    max_length = 22
    # dataset = MyDataset(df, 'datasets/dataset8', max_length=max_length)


    # x, y = dataset[0]
    # print(y)
    # print(x)
    # print(x.sum())

    dataset_rootdir = "datasets/dataset8"
    df_train, df_test = load_df(dataset_rootdir=dataset_rootdir, nrows=None)
    train_dataset = MyDataset(dataset_rootdir, df_train, max_length)
    val_dataset = MyDataset(dataset_rootdir, df_test, max_length)

    for i in tqdm(range(len(train_dataset))):
        x, y = train_dataset[i]

    