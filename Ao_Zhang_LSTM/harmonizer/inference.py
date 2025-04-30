#!/usr/bin/env python3
"""
Batch harmonization with Beam Search + Pitch-Continuity Smoothing:
Process all melody MIDIs in an input directory,
harmonize using a trained LSTMHarmonizer, apply beam search,
and smooth the chosen pitch by minimizing jumps between frames.
"""
import os
import argparse
from pathlib import Path
import numpy as np
import pretty_midi
import torch
from model import LSTMHarmonizer, EMBED_DIM, LSTM_HIDDEN, VOCAB_SIZE, NUM_VOICES


def midi_to_roll(path: Path, fs: int) -> np.ndarray:
    pm = pretty_midi.PrettyMIDI(str(path))
    pr = pm.get_piano_roll(fs=fs)
    return (pr > 0).astype(np.uint8).T


def beam_decode(logits: np.ndarray, beam_width: int = 5) -> np.ndarray:
    import heapq
    T, V = logits.shape
    heap = [(-logits[0, i], [int(i)]) for i in np.argsort(logits[0])[-beam_width:]]
    for t in range(1, T):
        new_heap = []
        topk = np.argsort(logits[t])[-beam_width:]
        for neg_score, seq in heap:
            for idx in topk:
                new_score = neg_score - logits[t, idx]
                new_heap.append((new_score, seq + [int(idx)]))
        heap = heapq.nsmallest(beam_width, new_heap, key=lambda x: x[0])
    best_seq = min(heap, key=lambda x: x[0])[1]
    return np.array(best_seq, dtype=np.int64)


def smooth_sequence(seqs: np.ndarray) -> np.ndarray:
    # seqs: (T,) array of top-k best candidates placeholder
    # Here seqs is from beam_decode; this function enforces continuity
    best = []
    prev = None
    for p in seqs:
        if prev is None:
            choice = p
        else:
            # for beam, p can be array of k, but here p is single seq
            # modify to assume p is 1D array of k candidates? Instead, integrate smoothing in main
            choice = p
        best.append(choice)
        prev = choice
    return np.array(best, dtype=np.int64)


def continuity_smooth(topk_preds: np.ndarray) -> np.ndarray:
    # topk_preds: (T, k) each row is k candidate pitches
    T, k = topk_preds.shape
    best_seq = []
    prev = None
    for t in range(T):
        candidates = topk_preds[t]
        if prev is None:
            # pick highest probability candidate (last one)
            choice = candidates[-1]
        else:
            # pick candidate minimizing absolute jump
            choice = int(min(candidates, key=lambda c: abs(c - prev)))
        best_seq.append(choice)
        prev = choice
    return np.array(best_seq, dtype=np.int64)


def roll_to_midi(mel, alto, tenor, bass, fs: int, out_path: Path):
    pm = pretty_midi.PrettyMIDI()
    for roll, program in zip([mel, alto, tenor, bass], [0,61,57,58]):
        inst = pretty_midi.Instrument(program=program)
        T = roll.shape[0]
        for pitch in range(128):
            on = False
            start = None
            for t in range(T):
                if roll[t,pitch] and not on:
                    on = True
                    start = t
                if on and (t==T-1 or not roll[t,pitch]):
                    end = t
                    inst.notes.append(
                        pretty_midi.Note(100, pitch, start/fs*0.25, end/fs*0.25)
                    )
                    on = False
        pm.instruments.append(inst)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMHarmonizer(input_dim=EMBED_DIM,
                           hidden_dim=LSTM_HIDDEN,
                           output_dim=VOCAB_SIZE).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    midi_paths = list(in_dir.rglob('*.mid'))
    print(f"Found {len(midi_paths)} MIDIs in {in_dir}")

    for midi_path in midi_paths:
        rel = midi_path.relative_to(in_dir)
        out_path = out_dir / rel.with_suffix('.mid')

        mel_roll = midi_to_roll(midi_path, args.fs)
        T = mel_roll.shape[0]
        with torch.no_grad():
            x = torch.from_numpy(mel_roll.astype(np.float32)).unsqueeze(0).to(device)
            outs = model(x)
        # generate each voice
        voices = []
        for head_out in outs:
            logits = head_out.squeeze(0).cpu().numpy()  # (T,V)
            # obtain top-k candidate indices per frame
            topk_indices = np.argsort(logits, axis=1)[:, -args.beam_width:]
            # apply continuity smoothing on top-k
            seq = continuity_smooth(topk_indices)
            roll = np.zeros((T,128), dtype=np.uint8)
            for t,p in enumerate(seq): roll[t,p]=1
            voices.append(roll)
        alto_roll, tenor_roll, bass_roll = voices
        roll_to_midi(mel_roll, alto_roll, tenor_roll, bass_roll, args.fs, out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',  type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--ckpt',       type=str, default='lstm_best.pth')
    parser.add_argument('--fs',         type=int, default=8)
    parser.add_argument('--beam-width', type=int, default=5)
    args = parser.parse_args()
    main(args)
