import torch
from torch import nn
from torch.nn.functional import softmax


class MultiHeadAttention(nn.Module):
    def __init__(self, num_attention_heads=8) -> None:
        super().__init__()
        self.linear_q = nn.Linear(64, 64)
        self.linear_k = nn.Linear(64, 64)
        self.linear_v = nn.Linear(64, 64)
        self.attention_output = nn.Linear(num_attention_heads * 64, 512)
        self.num_attention_heads = num_attention_heads
        self.dummy_param = nn.Parameter(
            torch.sqrt(torch.Tensor([64])), requires_grad=False
        )
        pass

    def forward(self, query, key, value, mask=None):
        bs = key.shape[0]
        single_embed_size = int(key.shape[-1] / self.num_attention_heads)
        tokens = key.shape[1]
        query_tokens = query.shape[1]
        # Divide the data for multiple heads
        query = query.reshape(
            bs, query_tokens, self.num_attention_heads, single_embed_size
        )
        key = key.reshape(bs, tokens, self.num_attention_heads, single_embed_size)
        value = value.reshape(bs, tokens, self.num_attention_heads, single_embed_size)

        # Project the data using respective linear layers
        q = torch.einsum("ijkl -> ikjl", [self.linear_q(query)])
        k = torch.einsum("ijkl -> ikjl", [self.linear_k(key)])
        v = torch.einsum("ijkl -> ikjl", [self.linear_v(value)])

        # Compute scaled dot product attention
        attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # Concatenate and return the output
        concatenated_weights = attention_weights.reshape(bs, query_tokens, -1)
        out = self.attention_output(concatenated_weights)
        return out

    def scaled_dot_product_attention(self, q, k, v, mask):
        # Compute the dot product
        dot_product = torch.einsum("...kl, ...rl -> ...kr", [q, k])

        if mask is not None:
            dot_product = dot_product.masked_fill(mask == 0, 1e-9)
        # Scale the dot product
        dot_product = dot_product / self.dummy_param
        # Compute the softmax and multiply with values
        attention_weights = torch.einsum(
            "ijkl, ijlr -> ikjr", [softmax(dot_product, dim=-1), v]
        )
        return attention_weights
