"""Test the encoder and decoder forward passes."""

import torch

from src.Transformer.model.decoder import Decoder
from src.Transformer.model.encoder import Encoder
from src.Transformer.model.model import Transformer


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


def test_model():
    """Test the transformer model."""
    src_vocab_size = 100
    tgt_vocab_size = 100
    batch_size = 32
    n_tokens = 10
    encoder_inputs = torch.randint(99, (batch_size, n_tokens))
    decoder_inputs = torch.randint(99, (batch_size, n_tokens))
    transformer = Transformer(src_vocab_size, tgt_vocab_size)
    outputs = transformer(encoder_inputs, decoder_inputs)
    assert list(outputs.shape) == [batch_size, n_tokens, tgt_vocab_size]
