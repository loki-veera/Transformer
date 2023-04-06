"""Script to train the Transformer model."""
import torch
from tqdm import tqdm

from .dataloader import TextDataset
from .model.Transformer import Transformer


def train(model, device, loader, optimizer, epoch, loss):
    """Train the model for each epoch."""
    model.train()
    for data, target in tqdm(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        preds = model(data, target)
        target_onehot = (
            torch.nn.functional.one_hot(target, num_classes=loader.vocab_size)
            .type("torch.FloatTensor")
            .to(device)
        )
        loss_val = loss(preds, target_onehot)
        loss_val.backward()
        optimizer.step()
    print(
        "Train Epoch: {}\tLoss: {:.6f}".format(
            epoch,
            loss_val.item(),
        ),
        flush=True,
    )


def main():
    """Train the transformer to generate poems as shakespeare."""
    data_path = "data/tinyshakespeare"
    batch_size = 100
    seq_len = 30
    lr = 0.0015
    num_epochs = 5

    dataloader = TextDataset(data_path, batch_size, seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    src_size = tgt_size = dataloader.vocab_size
    model = Transformer(src_size, tgt_size, device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()

    params_no = sum([p.data.nelement() for p in model.parameters()])
    print(f"Total number of parameters in the network: {params_no}")

    for epoch in range(num_epochs):
        train(model, device, dataloader, optimizer, epoch, loss)

    model_name = "model.pt"
    torch.save(model.state_dict(), model_name)


if __name__ == "__main__":
    main()
