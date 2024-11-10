import torch
from torchvision.models import vit_b_16
from vit import VitEncoder
from dataset import MyDataset, load_df

# def evaluation():
#     evaluator.run(val_loader)
#     metrics = evaluator.state.metrics
#     avg_accuracy = metrics['accuracy']
#     loss = metrics['loss']
#     tqdm.write(
#         f'Validation Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.2f} Avg loss: {loss:.2f}'
#     )    

def eval_sample(dataset_rootdir='datasets/dataset6'):
    max_lenth = 10
    vit = vit_b_16(weights='IMAGENET1K_V1')
    model = VitEncoder(vit, max_lenth)    
    model.cuda()

    pt = torch.load('checkpoints/model_22.pt')

    model.load_state_dict(pt)
    model.eval()

    _df_train, df_test = load_df(dataset_rootdir, nrows=None)
    val_dataset = MyDataset(dataset_rootdir, df_test, max_lenth)

    x, y = val_dataset[1]
    x = x.cuda()

    y_pred = model(x.unsqueeze(0))
    y_pred = y_pred.argmax(-1)

    true = val_dataset.tokenizer.decode(y[0].tolist())
    pred = val_dataset.tokenizer.decode(y_pred[0].tolist())
    print('true: ', true)
    print('pred: ', pred)



if __name__ == '__main__':
    eval_sample()
