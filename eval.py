import torch
from torchvision.models import vit_b_16
from vit import VitEncoder
from dataset import MyDataset, load_df


def main(dataset_rootdir='dataset2'):
    vit = vit_b_16(weights='IMAGENET1K_V1')
    model = VitEncoder(vit)    
    pt = torch.load('checkpoints/model_1.pt')

    model.load_state_dict(pt)
    model.eval()

    _df_train, df_test = load_df(dataset_rootdir, nrows=100)
    val_dataset = MyDataset(dataset_rootdir, df_test)

    x, y = val_dataset[0]

    y_pred = model(x.unsqueeze(0))
    y_pred = y_pred.argmax(-1)

    true = val_dataset.tokenizer.decode(y[0].tolist())
    pred = val_dataset.tokenizer.decode(y_pred[0].tolist())
    print('true: ', true)
    print('pred: ', pred)



if __name__ == '__main__':
    main()
