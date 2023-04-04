from torch import nn
import torch

class ComputeEmbeddings(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 512)

    def forward(self, inputs):
        input_embeds = self.embed(inputs)
        return input_embeds + self.positional_encoding()

    def positional_encoding(self):
        max_seq_length = 5000
        output_dim = 512
        pos_matrix = torch.zeros(max_seq_length,
                                 output_dim, 
                                 requires_grad=False
                                )
        for seq_index in range(0, max_seq_length):
            for index in range(0, output_dim // 2):
                denom = seq_index/(10000**((2*index)/output_dim))
                (pos_matrix[seq_index, 2*index],
                 pos_matrix[seq_index, 2*index + 1]) = torch.sin(denom), torch.cos(denom)
        return pos_matrix

