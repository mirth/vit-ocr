import torch
import pandas as pd
from torchvision.models import vit_b_16
from vit import VitEncoder
from dataset import MyDataset, load_df
from dataset_iiit5k import DatasetIIIT5K
import tokenizer

# def evaluation():
#     evaluator.run(val_loader)
#     metrics = evaluator.state.metrics
#     avg_accuracy = metrics['accuracy']
#     loss = metrics['loss']
#     tqdm.write(
#         f'Validation Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.2f} Avg loss: {loss:.2f}'
#     )    

def eval_sample(dataset_rootdir='datasets/IIIT5K-Word_V3.0/IIIT5K'):
    max_lenth = 22
    vit = vit_b_16(weights='IMAGENET1K_V1')
    model = VitEncoder(vit, max_lenth)    
    model.cuda()

    # pt = torch.load('checkpoints/model_2.pt')
    pt = torch.load('archive/model_1_dataset6_acc=1.pt')

    model.load_state_dict(pt)
    model.eval()

    # _df_train, df_test = load_df(dataset_rootdir, nrows=None)
    # val_dataset = MyDataset(dataset_rootdir, df_test, max_lenth)
    df_test = pd.read_csv('datasets/IIIT5K-Word_V3.0/IIIT5K-Word_V3.0_IIIT5K_testdata.csv')
    val_dataset = DatasetIIIT5K(df_test, dataset_rootdir='datasets/IIIT5K-Word_V3.0/IIIT5K', max_length=max_lenth)

    x, y = val_dataset[2]
    x = x.cuda()

    y_pred = model(x.unsqueeze(0))
    y_pred = y_pred.argmax(-1)

    true = tokenizer.decode(y.tolist())
    pred = tokenizer.decode(y_pred[0].tolist())
    print('true: ', true)
    print('pred: ', pred)



if __name__ == '__main__':
    eval_sample()
