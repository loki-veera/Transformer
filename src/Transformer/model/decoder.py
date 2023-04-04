from torch import nn

class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Instantiate the decoder layer
        # Make N copies of the decoder
        pass

    def forward(self, encoder_outputs, outputs):
        # Pass through each layer of the decoder
        pass


class DecoderLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Instantiate the masked multi head attention here
        # Instantiate the multi head attention here
        # Instantiate the feed forward layer here
        pass

    def forward(self, encoder_outputs, outputs):
        # Pass the respective inputs through the above instantiated layers
        pass