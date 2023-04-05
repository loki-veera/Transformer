from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear_q = nn.Linear(512, 64)
        self.linear_k = nn.Linear(512, 64)
        self.linear_v = nn.Linear(512, 64)
        self.attention_output = nn.Linear(8*64, 512)
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