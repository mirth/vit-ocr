import torch
from torch import nn
from torch.optim import Adam
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader
from ignite.metrics import Accuracy, Loss
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.engine import Engine
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import RunningAverage
from ignite.utils import setup_logger
from ignite.handlers import Checkpoint
from tqdm import tqdm
from dataset import MyDataset
from attn import CrossAttention


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

VOCAB_SIZE = 50257 + 1
# def count_parameters(model):
#     for name, p in model.named_parameters():
#         print(name, p.numel())

class VitEncoder(nn.Module):
    def __init__(self, vit):
        super().__init__()

        vit.heads = nn.Identity()
        self.vit = vit
        self.cross_attn = CrossAttention(
            kv_dim=768,
            q_dim=VOCAB_SIZE,
            qk_out_dim=768,
            v_out_dim=768,
            num_heads=768 // 128,
        )
        self.latents = nn.Parameter(torch.randn(32, VOCAB_SIZE))

    def forward(self, x):
        n, c, h, w = x.shape

        x = self.vit._process_input(x)
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder(x)
        x = self.cross_attn(inputs_kv=x, inputs_q=self.latents.repeat(n, 1, 1))

        return x
    
def output_transform(output):
    y_pred, y = output
    y_pred = y_pred.view(-1, y_pred.size(-1))
    y = y.contiguous().view(-1)

    return y_pred, y

def load_df():
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    df = pd.read_csv("dataset0/data.csv")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    return df_train, df_test


def main():
    batch_size = 2
    lr = 1e-4
    epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vit = vit_b_16(weights='IMAGENET1K_V1')
    model = VitEncoder(vit)

    df_train, df_test = load_df()
    train_dataset = MyDataset(df_train)
    val_dataset = MyDataset(df_test)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

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



if __name__ == '__main__':
    main()

    # x = torch.randn(1, 3, 224, 224)
    # y = model(x)
    # print(count_parameters(model))
    # print(y.shape)
    # print(model)