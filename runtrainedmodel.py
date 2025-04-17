#!/usr/bin/env python
"""
Run a trained TransformerHarmonizer checkpoint on new soprano input.

Example:
    python run_harmonizer.py \
        --soprano /path/to/new_soprano.json \
        --checkpoint saved_models/transformer_harmonizer_epoch3.pt \
        --out_dir results

If you give --soprano a *folder* instead of a single file, every .json
inside it is processed.
"""
import argparse, os, json, torch, pretty_midi
from music21 import note, stream
from methon1 import (               # imports everything we need
    build_model, load_checkpoint, PAD_TOKEN, SOS_TOKEN, DEVICE
)

################################################################################
# CLI
################################################################################
parser = argparse.ArgumentParser(description="Generate SATB harmonization.")
parser.add_argument("--soprano", required=True,
                    help="Path to a soprano JSON file *or* a folder of JSONs.")
parser.add_argument("--checkpoint",
                    default="saved_models/transformer_harmonizer_epoch3.pt",
                    help="Model checkpoint to load (default: latest epoch3).")
parser.add_argument("--out_dir", default="run_outputs",
                    help="Directory where SATB JSON & MIDI are written.")
parser.add_argument("--temperature", type=float, default=1.2,
                    help="Sampling temperature (higher → more variety).")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

################################################################################
# Load checkpoint
################################################################################
model, epoch = load_checkpoint(args.checkpoint)
model.eval()

################################################################################
# Gather list of soprano JSON files
################################################################################
soprano_files = []
if os.path.isdir(args.soprano):
    soprano_files = [os.path.join(args.soprano, f)
                     for f in sorted(os.listdir(args.soprano))
                     if f.endswith(".json")]
else:
    soprano_files = [args.soprano]

assert soprano_files, "No .json files found for --soprano"

################################################################################
# Helper: one‑shot conversion to MIDI
################################################################################
def save_midi(satb, midi_path):
    parts = [stream.Part(id=n) for n in ("S", "A", "T", "B")]
    for chord in satb:
        for i, p in enumerate(chord):
            n = note.Note()
            n.pitch.midi = int(p)
            n.quarterLength = 1
            parts[i].append(n)
    score = stream.Score(parts)
    score.write("midi", fp=midi_path)

################################################################################
# Run each file
################################################################################
for sp_path in soprano_files:
    with open(sp_path) as f:
        soprano_seq = json.load(f)

    src = torch.tensor([soprano_seq], dtype=torch.long, device=DEVICE)
    gen_len = len(soprano_seq)

    with torch.no_grad():
        atb = model.sample_decode(src, max_len=gen_len,
                                  temperature=args.temperature)[0].cpu().tolist()

    satb = [[soprano_seq[t]] + atb[t] for t in range(gen_len)]

    # Write JSON
    base = os.path.splitext(os.path.basename(sp_path))[0]
    json_out = os.path.join(args.out_dir, f"{base}_satb.json")
    with open(json_out, "w") as f:
        json.dump(satb, f, indent=2)
    print(f"→ SATB JSON: {json_out}")

    # Write MIDI
    midi_out = os.path.join(args.out_dir, f"{base}_satb.mid")
    save_midi(satb, midi_out)
    print(f"→ MIDI:      {midi_out}\n")