from torch import nn
from multi_head_attention import MultiHeadAttention

class Decoder(nn.Module):
    def __init__(self, N = 6) -> None:
        super().__init__()
        self.deocder = nn.ModuleList(
                [
                    DecoderLayer()
                    for _ in range(N)
                ])

    def forward(self, encoder_outputs, outputs):
        for decoder_layer in self.deocder:
            outputs = decoder_layer(encoder_outputs, outputs)
        return outputs


class DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.masked_attention = MultiHeadAttention()
        self.decoder_attention = MultiHeadAttention()
        self.decoder_linear = nn.Linear(512, 2048)
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, encoder_outputs, outputs):
        stage_one_output = self.layer_norm(
                            outputs + 
                            self.masked_attention(outputs,
                                                  outputs,
                                                  outputs
                                                )
                            )
        stage_two_output = self.layer_norm(
                            stage_one_output +
                            self.decoder_attention(encoder_outputs,
                                                   encoder_outputs,
                                                   stage_two_output
                                                   )
                            )
        stage_three_output = self.layer_norm(
                                stage_two_output +
                                self.decoder_linear(stage_two_output)
                                )
        return stage_three_output
