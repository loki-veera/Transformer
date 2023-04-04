from torch import nn

class ComputeEmbeddings(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Instantiate the nn.Embedding module here
        pass

    def forward(self, inputs):
        # Compute the input embeddings
        # Compute the positional embeddings
        # Concatente both the embeddings
        pass