import torch
from torch import nn
from torch.optim import Adam
from torchvision.models import vit_b_16
from ignite.metrics import Accuracy, Loss
from ignite.engine import Events, create_supervised_trainer
from ignite.engine import Engine
from ignite.contrib.handlers import ProgressBar
from torch.utils.data import DataLoader
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
        print(x.shape)
        x = self.cross_attn(inputs_kv=x, inputs_q=self.latents.repeat(n, 1, 1))

        return x
    

def main():
    batch_size = 2
    lr = 1e-4
    epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vit = vit_b_16(weights='IMAGENET1K_V1')
    model = VitEncoder(vit)

    train_dataset = MyDataset()
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    # val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    val_metrics = {"accuracy": Accuracy(), "less": Loss(criterion)}
    # trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    def train_step(engine, batch):
        model.train()
        
        x, y = batch
        x = x.to(device)
        y = y.to(device)    

        y_hat = model(x)

        y_hat = y_hat.view(-1, y_hat.size(-1))
        y = y.contiguous().view(-1)

        loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    trainer = Engine(train_step)

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=["loss"])
    # @trainer.on(Events.ITERATION_COMPLETED(every=1))
    # def log_training_loss(engine):
    #     pbar.desc = f"ITERATION - loss: {engine.state.output:.2f}"
    #     pbar.update(log_interval)
    
    trainer.run(train_loader, max_epochs=epochs)

    # x = torch.randn(1, 3, 224, 224)
    # y = model(x)
    # print(count_parameters(model))
    # print(y.shape)
    # print(model)


if __name__ == '__main__':
    main()