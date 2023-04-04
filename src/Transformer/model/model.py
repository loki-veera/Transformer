from torch import nn
from embeddings import ComputeEmbeddings
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size) -> None:
        super().__init__()
        self.input_embeddings = ComputeEmbeddings(src_vocab_size)
        self.output_embeddings = ComputeEmbeddings(tgt_vocab_size)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.linear_layer = nn.Linear(512, tgt_vocab_size)  # Confirm the output size
    
    def forward(self, inputs, outputs):
        embed_inputs = self.input_embeddings(inputs)
        embed_outputs = self.output_embeddings(outputs)

        encoder_outputs = self.encoder(embed_inputs)
        decoder_outputs = self.decoder(encoder_outputs, embed_outputs)
        
        output = self.linear_layer(decoder_outputs)
        return nn.Softmax(output)