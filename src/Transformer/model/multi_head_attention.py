from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Instantitate the 4 linear layers (for Q, K, V and output Linear)
        pass

    def forward(self, query, key, value):
        # Compute the projections of query, key and value with the respective linear layers
        # Compute the scaled dot product attention
        # Concatenate the results
        # Project the concatenated results with final linear layer
        pass

    def scaled_dot_product_attention(self, Q, K, V):
        # Implement the softmax equation 1 in Attention is all you need paper.
        # https://arxiv.org/pdf/1706.03762.pdf
        pass