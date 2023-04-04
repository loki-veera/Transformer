from torch import nn

class Transformer(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # Initate the input embeddings here
        # Initate the positional embeddings here
        # Initiate the encoder here
        # Initiate the decoder here
        # Initiate the end linear layer here
        pass
    
    def forward(self, inputs, outputs):
        # Pass the x and y through the embeddings and add the position embeddings
        # Pass the results through the encoder
        # pass the outputs and the encoder output through the decoder
        pass