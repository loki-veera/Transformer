from torch import nn

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Instantiate the encoder layer
        # Make N copies of the encoder
        pass

    def forward(self, inputs):
        # Pass through each layer of the encoder
        pass


class EncoderLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Instantiate the multi head attention here
        # Instantiate the feed forward layer here
        pass

    def forward(self, inputs):
        # Pass the respective inputs through the above instantiated layers
        pass