import math
import torch
from torch import nn
from torch.optim import Adam
from torchvision.models import vit_b_16
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from ignite.metrics import Accuracy, Loss
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.engine import Engine
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import RunningAverage
from ignite.utils import setup_logger
from ignite.handlers import Checkpoint
from ignite.handlers.param_scheduler import LRScheduler
from tqdm import tqdm
from dataset import MyDataset, load_df
from vit import VitEncoder


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def output_transform(output):
    y_pred, y = output
    y_pred = y_pred.view(-1, y_pred.size(-1))
    y = y.contiguous().view(-1)

    return y_pred, y


def main(dataset_rootdir='datasets/dataset6'):
    batch_size = 6
    lr = 1e-4
    epochs = 100
    device = 'cuda'# if torch.cuda.is_available() else 'cpu'
    max_length = 10

    vit = vit_b_16(weights='IMAGENET1K_V1')
    model = VitEncoder(vit, max_length)

    # pt = torch.load('archive/model_9_dataset7_acc_0.9.pt')
    # model.load_state_dict(pt)
    model.cuda()

    df_train, df_test = load_df(dataset_rootdir=dataset_rootdir, nrows=None)
    train_dataset = MyDataset(dataset_rootdir, df_train, max_length)
    val_dataset = MyDataset(dataset_rootdir, df_test, max_length)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    T_max = math.ceil(len(train_dataset) / batch_size)

    torch_lr_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0.00001)
    lr_scheduler = LRScheduler(torch_lr_scheduler)    

    def train_step(engine, batch):
        model.train()
        
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)

        y_pred = y_pred.view(-1, y_pred.size(-1))
        y = y.contiguous().view(-1)

        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def val_step(engine, batch):
        model.eval()
        optimizer.zero_grad()

        x, y = batch
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)

        y_pred = y_pred.view(-1, y_pred.size(-1))
        y = y.contiguous().view(-1)

        # loss = criterion(y_pred, y)

        return y_pred, y

    trainer = Engine(train_step)
    trainer.logger = setup_logger('trainer')
    trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

    @trainer.on(Events.EPOCH_STARTED)
    def print_lr():
        print('EPOCH_STARTED: ', optimizer.param_groups[0]["lr"])

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_lr():
        print('EPOCH_COMPLETED: ', optimizer.param_groups[0]["lr"])

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    pbar_train = ProgressBar()
    pbar_train.attach(trainer, metric_names=['loss'])

    evaluator = Engine(val_step)
    Loss(criterion, output_transform=output_transform).attach(evaluator, 'loss')
    Accuracy(output_transform=output_transform).attach(evaluator, 'accuracy')

    evaluator.logger = setup_logger('evaluator')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        loss = metrics['loss']
        tqdm.write(
            f'Validation Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.2f} Avg loss: {loss:.2f}'
        )

    to_save = {'model': model}
    gst = lambda *_: trainer.state.epoch
    handler = Checkpoint(
        to_save, './checkpoints', n_saved=3, global_step_transform=gst
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

    trainer.run(train_loader, max_epochs=epochs)

    # if evaluation:
    #     pt = torch.load('checkpoints/model_21.pt')
    #     model.load_state_dict(pt)
    #     model.eval()
    #     log_validation_results(evaluator)

if __name__ == '__main__':
    main()

    # x = torch.randn(1, 3, 224, 224)
    # y = model(x)
    # print(count_parameters(model))
    # print(y.shape)
    # print(model)