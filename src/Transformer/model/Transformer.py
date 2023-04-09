import torch
from torch import nn

from .decoder import Decoder
from .embeddings import ComputeEmbeddings
from .encoder import Encoder


class Transformer(nn.Module):
    def __init__(
        self, src_vocab_size, tgt_vocab_size, device, d_model=512, d_k=64
    ) -> None:
        super().__init__()
        self.input_embeddings = ComputeEmbeddings(src_vocab_size, d_model)
        self.output_embeddings = ComputeEmbeddings(tgt_vocab_size, d_model)
        self.encoder = Encoder(d_k, d_model)
        self.decoder = Decoder(d_k, d_model)
        self.linear_layer = nn.Linear(d_model, tgt_vocab_size)
        self.device = device

    def forward(self, inputs, outputs):
        embed_inputs, embed_outputs = self.__compute_embeddings(inputs, outputs)
        encoder_outputs = self.encoder(embed_inputs)
        mask = self.__decoder_mask(embed_outputs)
        decoder_outputs = self.decoder(encoder_outputs, embed_outputs, mask)

        output = self.linear_layer(decoder_outputs)
        return output

    def __decoder_mask(self, output_embeds):
        mask = torch.tril(
            torch.ones(output_embeds.shape[1], output_embeds.shape[1])
        ).expand(
            output_embeds.shape[0], 1, output_embeds.shape[1], output_embeds.shape[1]
        )
        return mask.to(self.device)

    def __compute_embeddings(self, inputs, outputs):
        embed_inputs = self.input_embeddings(inputs)
        embed_outputs = self.output_embeddings(outputs)
        return embed_inputs, embed_outputs
