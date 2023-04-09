"""Script to train the Transformer model."""
import time

import torch
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from tqdm import tqdm

from .dataloader import DatasetTransforms
from .model.Transformer import Transformer

torch.manual_seed(0)


def train(model, optimizer, loader, criterion, device):
    """Training loop."""
    model.train()
    total_loss = 0

    for src, tgt in tqdm(loader, total=len(list(loader))):
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()
        preds = model(src, tgt)
        loss = criterion(preds.reshape(-1, preds.shape[-1]), tgt.reshape(-1))
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(list(loader))


def main():
    """Trainnig and evaluation."""
    src_lang = "de"
    tgt_lang = "en"
    batch_size = 128
    num_epochs = 20
    lr = 0.0001
    data_transforms = DatasetTransforms(src_lang, tgt_lang)
    data_transforms.generate_transforms()
    data_transforms.apply_transforms()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_vocab_size = len(data_transforms.vocab_transform[src_lang])
    tgt_vocab_size = len(data_transforms.vocab_transform[tgt_lang])

    model = Transformer(src_vocab_size, tgt_vocab_size, device)
    for params in model.parameters():
        if params.dim() > 1:
            torch.nn.init.xavier_uniform_(params)
    model = model.to(device)
    print(f"Device: {device}")
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=data_transforms.PAD_IDX
    )  # We ignore the padding index
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    train_iter = Multi30k(split="train", language_pair=(src_lang, tgt_lang))
    train_loader = DataLoader(
        train_iter, batch_size=batch_size, collate_fn=data_transforms.collate_fn
    )

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        start = time.time()
        train_loss = train(model, optimizer, train_loader, criterion, device)
        end = time.time()
        print(f"Training loss: {train_loss}, Elapsed time: {end - start}")

    fname = "model.pt"
    torch.save(model.state_dict(), fname)


if __name__ == "__main__":
    main()
