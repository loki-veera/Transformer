"""Test the embeddings."""

import torch

from src.Transformer.model.embeddings import ComputeEmbeddings


# def test_embeddings():
#     """Test embeddings."""
#     torch.manual_seed(0)
#     inputs = torch.randn(32, 10, 512)
#     embed = ComputeEmbeddings()
#     embeddings = embed(inputs)
#     assert embeddings.shape == inputs.shape