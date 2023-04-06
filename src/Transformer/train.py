"""Script to train the Transformer model."""
from .dataloader import TextDataset
from .model.model import Transformer
import torch
from tqdm import tqdm

def train(model, device, loader, optimizer, epoch, loss):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(loader)):
        data, target =  data.to(device), target.to(device)
        optimizer.zero_grad()
        preds= model(data, target)
        target_onehot = torch.nn.functional.one_hot(target, num_classes=loader.vocab_size).type('torch.FloatTensor').to(device)
        loss_val = loss(preds, target_onehot)
        loss_val.backward()
        optimizer.step()
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(loader),
            100. * batch_idx / len(loader), loss_val.item()), flush=True)


def main():
    data_path = "data/tinyshakespeare"
    batch_size = 100
    seq_len = 30
    lr = 0.0015
    num_epochs = 5

    dataloader = TextDataset(data_path, batch_size, seq_len)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    src_size = tgt_size = dataloader.vocab_size
    model = Transformer(src_size, tgt_size, device) ## Fill the args later
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()

    params_no = sum([p.data.nelement() for p in model.parameters()])
    print(f"Total number of parameters in the network: {params_no}")

    for epoch in range(num_epochs):
        train(model, device, dataloader, optimizer, epoch, loss)

    model_name = 'model.pt'
    torch.save(model.state_dict(), model_name)

if __name__ == "__main__":
    main()