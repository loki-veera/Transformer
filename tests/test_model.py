"""Test the encoder and decoder forward passes."""

import torch

from src.Transformer.model.decoder import Decoder
from src.Transformer.model.encoder import Encoder


def test_encoder():
    """Test encoder forward pass using shapes."""
    torch.manual_seed(0)
    inputs = torch.randn(32, 10, 512)
    encoder = Encoder()
    encoder_output = encoder(inputs)
    assert list(encoder_output.shape) == [32, 10, 512]


def test_decoder():
    """Test decoder forward pass using shapes."""
    encoder_outputs = torch.randn(32, 10, 512)
    decoder_inputs = torch.randn(32, 10, 512)
    decoder = Decoder()
    decoder_output = decoder(encoder_outputs, decoder_inputs)
    assert list(decoder_output.shape) == [32, 10, 512]
