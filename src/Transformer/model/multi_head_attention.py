from torch import nn
import torch
from torch.nn.functional import softmax

class MultiHeadAttention(nn.Module):
    def __init__(self, num_attention_heads = 8) -> None:
        super().__init__()
        self.linear_q = nn.Linear(64, 64)
        self.linear_k = nn.Linear(64, 64)
        self.linear_v = nn.Linear(64, 64)
        self.attention_output = nn.Linear(num_attention_heads*64, 512)
        self.num_attention_heads = num_attention_heads
        pass

    def forward(self, query, key, value):
        bs = key.shape[0]
        single_embed_size = int(key.shape[-1]/self.num_attention_heads)
        tokens = key.shape[1]
        
        # Divide the data for multiple heads
        query = query.reshape(bs, tokens, self.num_attention_heads, single_embed_size)
        key   = key.reshape(bs, tokens, self.num_attention_heads, single_embed_size)
        value = value.reshape(bs, tokens, self.num_attention_heads, single_embed_size)
        
        # Project the data using respective linear layers
        Q = torch.einsum('ijkl -> ikjl',[self.linear_q(query)])
        K = torch.einsum('ijkl -> ikjl',[self.linear_k(key)])
        V = torch.einsum('ijkl -> ikjl',[self.linear_v(value)])

        # Compute scaled dot product attention
        attention_weights = self.scaled_dot_product_attention(Q, K, V)

        # Concatenate and return the output
        concatenated_weights = attention_weights.reshape(bs, tokens, -1)
        out = self.attention_output(concatenated_weights)
        return out

    def scaled_dot_product_attention(self, Q, K, V):
        # Compute the dot product
        dot_product = torch.einsum('...kl, ...rl -> ...kr',[Q, K])
        # Scale the dot product
        dot_product = dot_product/torch.sqrt(torch.Tensor([64]))
        # Compute the softmax and multiply with values
        attention_weights = torch.einsum('ijkl, ijlr -> ikjr', [softmax(dot_product, dim=-1), V])
        return attention_weights