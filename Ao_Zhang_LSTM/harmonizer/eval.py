#!/usr/bin/env python3
# eval.py

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import LSTMHarmonizer, EMBED_DIM, LSTM_HIDDEN, VOCAB_SIZE, NUM_VOICES

class CocoTinyDataset(Dataset):
    """
    Load *_coco_tiny.npz and filter out empty samples.
    Returns mel-roll (T,128) and target (3,T)
    """
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.mels   = data['mel']
        self.altos  = data['alto']
        self.tenors = data['tenor']
        self.basses = data['bass']
        self.samples = [i for i in range(len(self.mels)) if self.mels[i].shape[0] > 0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i = self.samples[idx]
        mel   = torch.from_numpy(self.mels[i].astype(np.float32))       # (T,128)
        alto  = torch.from_numpy(np.argmax(self.altos[i], axis=1).astype(np.int64))
        tenor = torch.from_numpy(np.argmax(self.tenors[i], axis=1).astype(np.int64))
        bass  = torch.from_numpy(np.argmax(self.basses[i], axis=1).astype(np.int64))
        target = torch.stack([alto, tenor, bass], dim=0)                # (3,T)
        return mel, target


def evaluate(model, loader, device, topk=(1,5)):
    """
    Evaluate model on loader, returning:
      - avg_loss_per_voice: list of CE loss per token
      - ppl_per_voice      : list of perplexity per voice
      - acc1_per_voice     : top-1 accuracy per voice
      - acck_per_voice     : top-K accuracy per voice (if k>1)
      - overall_acc1       : aggregated top-1 accuracy
      - overall_acck       : aggregated top-K accuracy
    """
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss = [0.0] * NUM_VOICES
    total_tokens = 0
    correct1 = [0] * NUM_VOICES
    correctk = {k: [0] * NUM_VOICES for k in topk if k > 1}

    model.eval()
    with torch.no_grad():
        for mel, target in loader:
            # mel: (B, T, 128), target: (B, 3, T)
            mel = mel.to(device)
            target = target.to(device)
            outputs = model(mel)  # list of 3: each (B, T, V)
            B, T, _ = mel.size()
            total_tokens += B * T

            for i, head_out in enumerate(outputs):
                # head_out: (B, T, V)
                logits = head_out.view(-1, VOCAB_SIZE)          # (B*T, V)
                gt = target[:, i, :].reshape(-1)               # (B*T,)
                total_loss[i] += criterion(logits, gt).item()

                # top-1
                preds1 = logits.argmax(dim=-1)                 # (B*T,)
                correct1[i] += (preds1 == gt).sum().item()

                # top-k
                for k in topk:
                    if k > 1:
                        topk_preds = logits.topk(k, dim=-1).indices  # (B*T, k)
                        matches = (topk_preds == gt.unsqueeze(-1))   # (B*T, k) bool
                        correctk[k][i] += matches.any(dim=-1).sum().item()

    # compute metrics
    avg_loss_per_voice = [l / total_tokens for l in total_loss]
    ppl_per_voice = [float(np.exp(l)) for l in avg_loss_per_voice]
    acc1_per_voice = [c / total_tokens for c in correct1]
    acck_per_voice = {k: [correctk[k][i] / total_tokens for i in range(NUM_VOICES)]
                      for k in correctk}
    overall_acc1 = sum(correct1) / (total_tokens * NUM_VOICES)
    overall_acck = {k: sum(correctk[k]) / (total_tokens * NUM_VOICES) for k in correctk}

    return (avg_loss_per_voice, ppl_per_voice,
            acc1_per_voice, acck_per_voice,
            overall_acc1, overall_acck)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LSTM Harmonizer with extended metrics")
    parser.add_argument('--ckpt',      required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument('--data_dir',  default='/Volumes/Kid_Rage 1/[_Database_]/CocoChorales/cocochorales_tiny_v1_midi')
    parser.add_argument('--batch_size',type=int, default=1)
    args = parser.parse_args()

    test_npz = os.path.join(args.data_dir, 'test_coco_tiny.npz')
    if not os.path.exists(test_npz):
        raise FileNotFoundError(f"Test file not found: {test_npz}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_ds = CocoTinyDataset(test_npz)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = LSTMHarmonizer(input_dim=EMBED_DIM,
                           hidden_dim=LSTM_HIDDEN,
                           output_dim=VOCAB_SIZE).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)

    (avg_loss, ppl, acc1, acck,
     overall_acc1, overall_acck) = evaluate(model, test_loader, device, topk=(1,5))

    print("=== Per-Voice CE Loss ===")
    for i, l in enumerate(avg_loss): print(f"Voice {i}: {l:.4f}")
    print("=== Per-Voice Perplexity ===")
    for i, p in enumerate(ppl): print(f"Voice {i}: {p:.2f}")
    print("=== Per-Voice Top-1 Accuracy ===")
    for i, a in enumerate(acc1): print(f"Voice {i}: {a:.4f}")
    print("=== Per-Voice Top-5 Accuracy ===")
    for i, a in enumerate(acck[5]): print(f"Voice {i}: {a:.4f}")
    print(f"Overall Top-1 Accuracy: {overall_acc1:.4f}")
    print(f"Overall Top-5 Accuracy: {overall_acck[5]:.4f}")

if __name__ == '__main__':
    main()
