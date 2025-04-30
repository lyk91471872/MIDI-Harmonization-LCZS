import torch.nn as nn

EMBED_DIM = 128
LSTM_HIDDEN = 256
NUM_VOICES = 3   # Alto, Tenor, Bass
VOCAB_SIZE = 128 # MIDI pitch range

class LSTMHarmonizer(nn.Module):
    def __init__(self, input_dim=EMBED_DIM, hidden_dim=LSTM_HIDDEN, output_dim=VOCAB_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, output_dim) 
                                    for _ in range(NUM_VOICES)])

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        # 返回三组 logits，分别对应 Alto, Tenor, Bass
        return [head(out) for head in self.heads]
