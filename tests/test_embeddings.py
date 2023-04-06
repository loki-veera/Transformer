"""Test the embeddings."""

import torch

from src.Transformer.model.embeddings import ComputeEmbeddings


def test_embeddings():
    """Test embeddings."""
    torch.manual_seed(0)
    inputs = torch.randint(100, (32, 10))
    embed = ComputeEmbeddings(vocab_size=100)
    embeddings = embed(inputs)
    assert list(embeddings.shape) == [32, 10, 512]
