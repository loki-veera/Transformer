"""Encoder definition."""

from torch import nn

from .multi_head_attention import MultiHeadAttention


class Encoder(nn.Module):
    """Encoder model."""

    def __init__(self, n=6) -> None:
        super().__init__()
        self.encoder = nn.ModuleList([EncoderLayer() for _ in range(n)])

    def forward(self, inputs):
        for encoder_layer in self.encoder:
            inputs = encoder_layer(inputs)
        return inputs


class EncoderLayer(nn.Module):
    """Each layer in encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder_attention = MultiHeadAttention()
        self.encoder_linear1 = nn.Linear(512, 2048)
        self.encoder_linear2 = nn.Linear(2048, 512)
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, inputs):
        stage_one_output = self.layer_norm(
            inputs + self.encoder_attention(inputs, inputs, inputs)
        )
        stage_two_output = self.layer_norm(
            stage_one_output + self.encoder_linear(stage_one_output)
        )
        return stage_two_output

    def encoder_linear(self, inputs):
        return self.encoder_linear2(self.encoder_linear1(inputs).relu())
