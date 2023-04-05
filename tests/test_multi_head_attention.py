"""
Test multi head attention
"""
import torch
from src.Transformer.model.multi_head_attention import MultiHeadAttention

def test_attention():
    """
    Test the outputs of Multi head attention
    """
    torch.manual_seed(0)
    query = torch.randn(32, 10, 512)
    key = torch.randn(32, 10, 512)
    value = torch.randn(32, 10, 512)
    attention = MultiHeadAttention()
    output = attention(query, key, value)
    assert list(output.shape) == [32, 10, 512]
