import torch
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import tokenizer


class DatasetIIIT5K:
    def __init__(self, df, *, dataset_rootdir, max_length):
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
        x, y = self.data[idx]

        x = Image.open(f'{self.dataset_rootdir}/' + x)
        if x.mode != 'RGB':
            x = x.convert('RGB')
        x = self.transform(x)

        y = y.lower()
        y = torch.from_numpy(np.array(tokenizer.encode(y, max_length=self.max_length)))

        return x, y