#!/usr/bin/env python
import os
import json
import random
import shutil
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import pretty_midi
import matplotlib.pyplot as plt
from music21 import note, stream

##############################
# Configuration and Paths
##############################
INPUT_FOLDER = "/Users/cui/Documents/GitHub/MIDI-Harmonization-LCZS/soprano_paragraphs"
OUTPUT_FOLDER = "/Users/cui/Documents/GitHub/MIDI-Harmonization-LCZS/atb_paragraphs"
MIDI_OUTPUT_FOLDER = "midi_outputs"
os.makedirs(MIDI_OUTPUT_FOLDER, exist_ok=True)
# Folder for predictions (.json) and generated sample MIDI
PRED_OUTPUT_FOLDER = "OUTPUT"
os.makedirs(PRED_OUTPUT_FOLDER, exist_ok=True)
# Folder for saving trained model checkpoints
MODEL_OUTPUT_FOLDER = "saved_models"
os.makedirs(MODEL_OUTPUT_FOLDER, exist_ok=True)

# ---------------------------
# Data-augmentation controls
# ---------------------------
AUGMENT_TRANSPOSITION = True        # flip to False during evaluation
TRANSP_RANGE = (-5, 6)              # inclusive semitone shift

##############################
# Global Hyperparameters (Reduced settings for limited memory)
##############################
HIDDEN_SIZE = 128            # Keep hidden size at 128
NUM_ENCODER_LAYERS = 2       # Reduced from 4 layers to 2 layers
NUM_DECODER_LAYERS = 2       # Reduced from 4 layers to 2 layers
NHEAD = 4                    # Use 4 attention heads instead of 8
DROPOUT = 0.1
PAD_TOKEN = 0
SOS_TOKEN = 128
INPUT_VOCAB_SIZE = 196
OUTPUT_VOCAB_SIZE = 196
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_model = HIDDEN_SIZE

##############################
# Positional Encoding Module
##############################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=DROPOUT, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

##############################
# Source Embedding (for S part)
##############################
class SourceEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(SourceEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=DROPOUT)
    def forward(self, x):
        # x: [batch, src_len]
        emb = self.embedding(x) * math.sqrt(d_model)
        emb = self.pos_encoder(emb)
        return emb  # [batch, src_len, d_model]

##############################
# Triple Embedding (for target ATB, each time step: [alto, tenor, bass])
##############################
class TripleEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TripleEmbedding, self).__init__()
        self.embed_alto = nn.Embedding(vocab_size, d_model)
        self.embed_tenor = nn.Embedding(vocab_size, d_model)
        self.embed_bass = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(3*d_model, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=DROPOUT)
    def forward(self, triple):
        # triple: [batch, trg_len, 3]
        token_alto = triple[:,:,0]
        token_tenor = triple[:,:,1]
        token_bass = triple[:,:,2]
        emb_a = self.embed_alto(token_alto)   # [batch, trg_len, d_model]
        emb_t = self.embed_tenor(token_tenor)
        emb_b = self.embed_bass(token_bass)
        cat_emb = torch.cat([emb_a, emb_t, emb_b], dim=2)  # [batch, trg_len, 3*d_model]
        proj_emb = self.proj(cat_emb)                      # [batch, trg_len, d_model]
        proj_emb = self.pos_encoder(proj_emb)
        return proj_emb

##############################
# TransformerHarmonizer Class with all modifications
##############################
class TransformerHarmonizer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead,
                 num_encoder_layers, num_decoder_layers, dropout=DROPOUT,
                 hold_steps: int = 2):
        super(TransformerHarmonizer, self).__init__()
        self.src_embedding = SourceEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TripleEmbedding(tgt_vocab_size, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dropout=dropout)
        self.fc_alto = nn.Linear(d_model, tgt_vocab_size)
        self.fc_tenor = nn.Linear(d_model, tgt_vocab_size)
        self.fc_bass = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model
        # Ensure we repeat each harmony token for at least `hold_steps`
        self.hold_steps = max(1, int(hold_steps))

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
        return mask

    # ------------------------------------------------------------------
    # Utility: mask logits so that no generated pitch exceeds soprano
    # ------------------------------------------------------------------
    def _mask_high_notes(self, logits: torch.Tensor, soprano_pitch: torch.Tensor) -> torch.Tensor:
        """
        Ensures that the model cannot choose a pitch higher than the current
        soprano pitch.
 
        Args:
            logits: Tensor of shape [batch, vocab_size] – raw logits for a part.
            soprano_pitch: Tensor of shape [batch] – MIDI pitches of soprano
                           for the current time‑step (0 for PAD).
 
        Returns:
            logits with positions > soprano_pitch set to -inf so their
            probability becomes zero after softmax.
        """
        # Skip masking on padded timesteps (soprano_pitch == 0)
        valid_mask = soprano_pitch > 0
        if valid_mask.any():
            batch, vocab = logits.shape
            pitch_range = torch.arange(vocab, device=logits.device).expand(batch, -1)
            soprano_exp = soprano_pitch.unsqueeze(1).expand_as(pitch_range)
            over_mask = (pitch_range > soprano_exp) & valid_mask.unsqueeze(1)
            logits = logits.masked_fill(over_mask, float("-inf"))
        return logits

    def forward(self, src, tgt, tgt_mask=None, teacher_forcing_ratio=1.0):
        src_emb = self.src_embedding(src).transpose(0, 1)  # [src_len, batch, d_model]
        memory = self.transformer.encoder(src_emb)
        batch_size, trg_len, _ = tgt.shape
        if teacher_forcing_ratio >= 1.0:
            SOS_triple = torch.full((batch_size, 1, 3), SOS_TOKEN, dtype=torch.long, device=tgt.device)
            tgt_input = torch.cat([SOS_triple, tgt[:, :-1, :]], dim=1)
            tgt_emb = self.tgt_embedding(tgt_input).transpose(0,1)
            if tgt_mask is None:
                tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(0)).to(tgt.device)
            output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask).transpose(0,1)
            alto_logits = self.fc_alto(output)
            tenor_logits = self.fc_tenor(output)
            bass_logits = self.fc_bass(output)
            return alto_logits, tenor_logits, bass_logits
        else:
            # Scheduled sampling branch.
            ys = torch.full((batch_size, 1, 3), SOS_TOKEN, dtype=torch.long, device=src.device)
            logits_alto_list = []
            logits_tenor_list = []
            logits_bass_list = []
            for t in range(trg_len):
                tgt_emb = self.tgt_embedding(ys).transpose(0,1)
                current_len = tgt_emb.size(0)
                current_mask = self.generate_square_subsequent_mask(current_len).to(src.device)
                out = self.transformer.decoder(tgt_emb, memory, tgt_mask=current_mask).transpose(0,1)
                last_out = out[:, -1, :]  # [batch, d_model]

                # Raw logits for loss computation
                curr_alto_raw  = self.fc_alto(last_out)
                curr_tenor_raw = self.fc_tenor(last_out)
                curr_bass_raw  = self.fc_bass(last_out)

                # Masked copy for *sampling only*
                soprano_pitch_now = src[:, t].to(curr_alto_raw.device)
                curr_alto_masked  = self._mask_high_notes(curr_alto_raw.clone(),  soprano_pitch_now)
                curr_tenor_masked = self._mask_high_notes(curr_tenor_raw.clone(), soprano_pitch_now)
                curr_bass_masked  = self._mask_high_notes(curr_bass_raw.clone(),  soprano_pitch_now)

                # Append *raw* logits for the loss
                logits_alto_list.append(curr_alto_raw.unsqueeze(1))
                logits_tenor_list.append(curr_tenor_raw.unsqueeze(1))
                logits_bass_list.append(curr_bass_raw.unsqueeze(1))
                if torch.rand(1).item() < teacher_forcing_ratio:
                    next_input = tgt[:, t:t+1, :]
                else:
                    pred_alto  = curr_alto_masked.argmax(dim=1, keepdim=True)
                    pred_tenor = curr_tenor_masked.argmax(dim=1, keepdim=True)
                    pred_bass  = curr_bass_masked.argmax(dim=1, keepdim=True)
                    next_input = torch.cat([pred_alto, pred_tenor, pred_bass], dim=1).unsqueeze(1)
                ys = torch.cat([ys, next_input], dim=1)
            alto_logits = torch.cat(logits_alto_list, dim=1)
            tenor_logits = torch.cat(logits_tenor_list, dim=1)
            bass_logits = torch.cat(logits_bass_list, dim=1)
            return alto_logits, tenor_logits, bass_logits

    def sample_decode(self, src, max_len, temperature=0.5, print_debug=False):
        src_emb = self.src_embedding(src).transpose(0,1)
        memory = self.transformer.encoder(src_emb)
        batch_size = src.size(0)
        ys = torch.full((batch_size, 1, 3), SOS_TOKEN, dtype=torch.long, device=src.device)
        if print_debug:
            print("Initial input (SOS):")
            print(ys)
        last_generated = ys[:, -1:, :]  # initialize with SOS, shape [batch,1,3]
        for t in range(max_len):
            if t % self.hold_steps == 0:
                tgt_emb = self.tgt_embedding(ys).transpose(0,1)
                current_len = tgt_emb.size(0)
                tgt_mask = self.generate_square_subsequent_mask(current_len).to(src.device)
                out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask).transpose(0,1)
                last_out = out[:, -1, :]  # [batch, d_model]

                logits_alto  = self.fc_alto(last_out)
                logits_tenor = self.fc_tenor(last_out)
                logits_bass  = self.fc_bass(last_out)

                # Mask logits so no harmony note is above the soprano
                soprano_pitch_now = src[:, t].to(logits_alto.device)
                logits_alto  = self._mask_high_notes(logits_alto,  soprano_pitch_now)
                logits_tenor = self._mask_high_notes(logits_tenor, soprano_pitch_now)
                logits_bass  = self._mask_high_notes(logits_bass,  soprano_pitch_now)

                prob_alto  = torch.softmax(logits_alto  / temperature, dim=1)
                prob_tenor = torch.softmax(logits_tenor / temperature, dim=1)
                prob_bass  = torch.softmax(logits_bass  / temperature, dim=1)

                pred_alto  = torch.multinomial(prob_alto,  num_samples=1)
                pred_tenor = torch.multinomial(prob_tenor, num_samples=1)
                pred_bass  = torch.multinomial(prob_bass,  num_samples=1)

                next_token = torch.cat([pred_alto, pred_tenor, pred_bass], dim=1).unsqueeze(1)
                last_generated = next_token  # update cache
            else:
                # Hold previous harmony note for minimum duration
                next_token = last_generated.clone()

            ys = torch.cat([ys, next_token], dim=1)

            if print_debug:
                print(f"Time step {t+1}:")
                print("Generated/held token:")
                print(next_token)
                print("Updated ys:")
                print(ys)
        return ys[:, 1:, :]  # Exclude initial SOS

##############################
# Build Model, Optimizer, Criterion (Reduced scale)
##############################
def build_model():
    # Set `hold_steps` to 2 for eighth‑notes (2×1/16) or 4 for quarter‑notes (4×1/16)
    model = TransformerHarmonizer(INPUT_VOCAB_SIZE, OUTPUT_VOCAB_SIZE, d_model, nhead=NHEAD,
                                  num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
                                  dropout=DROPOUT, hold_steps=4)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    return model, optimizer, criterion

##############################
# Model Checkpoint Helpers
##############################
def save_checkpoint(model, optimizer, epoch, path):
    """
    Save model + optimizer state for later fine‑tuning or inference.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved to '{path}'.")

def load_checkpoint(path, device=DEVICE):
    """
    Rebuild the model (with current hyper‑params) and load weights.
    Returns the model and the epoch number stored in the checkpoint.
    """
    checkpoint = torch.load(path, map_location=device)
    model, _, _ = build_model()          # uses current globals
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    epoch = checkpoint.get('epoch', 0)
    print(f"Loaded checkpoint from '{path}' (epoch {epoch}).")
    return model, epoch

##############################
# Consonance / Dissonance Helper
##############################
_CONSONANT_CLASSES = {0, 3, 4, 5, 7, 8, 9}  # pitch‑class intervals considered consonant
def dissonance_penalty(soprano: torch.Tensor, harmony: torch.Tensor, pad_val=PAD_TOKEN) -> torch.Tensor:
    """
    Compute average dissonance mask between soprano (batch, seq) and harmony (batch, seq).
    
    Returns a scalar tensor = proportion of non-pad positions that are dissonant.
    """
    # Mask out padded positions
    valid = (harmony != pad_val) & (soprano != pad_val)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=soprano.device)
    interval = torch.abs(harmony - soprano) % 12  # pitch‑class interval
    dissonant = ~torch.isin(interval, torch.tensor(list(_CONSONANT_CLASSES), device=soprano.device))
    return dissonant[valid].float().mean()

# ------------------------------------------------------------------
# Tonal‑theory chord‑legality helper
# ------------------------------------------------------------------
_ALLOWED_CHORD_STRUCTS = [
    {0, 4, 7},        # Major triad
    {0, 3, 7},        # Minor triad
    {0, 3, 6},        # Diminished triad
    {0, 4, 8},        # Augmented triad
    {0, 4, 7, 11},    # Major‑major 7th
    {0, 4, 7, 10},    # Dominant 7th
    {0, 3, 7, 10},    # Minor‑minor 7th
    {0, 3, 6, 10},    # Half‑diminished 7th
    {0, 3, 6, 9},     # Fully‑diminished 7th
]
# ------------------------------------------------------------------
# Progression helper: encourage V(7) → I or V ↔ V7
# ------------------------------------------------------------------
def dominant_progress_penalty(soprano, alto, tenor, bass, pad_val=PAD_TOKEN):
    """
    Return scalar tensor = fraction of dominant chords whose following chord
    *fails* to resolve to the tonic (down a perfect fifth) OR move between V
    and V7 (same root).
    
    Root is approximated by bass pitch‑class; this works because Bach chorales
    very rarely invert V or I outside 6‑4 cadential patterns.
    """
    batch, seq = soprano.shape
    viol = dom = 0
    for b in range(batch):
        for t in range(seq - 1):       # look ahead one step
            if soprano[b, t] == pad_val:
                continue
            # Build pitch‑class set of current chord (relative to bass root)
            root = int(bass[b, t] % 12)
            pcs  = {int(p % 12) for p in (soprano[b, t], alto[b, t], tenor[b, t], bass[b, t])}
            rel  = {(p - root) % 12 for p in pcs}
            is_dom_triad  = rel == {0, 4, 7}
            is_dom_seventh = rel == {0, 4, 7, 10}
            if not (is_dom_triad or is_dom_seventh):
                continue
            dom += 1
            next_root = int(bass[b, t + 1] % 12)
            semitone_down_fifth = (root - 7) % 12  # resolve to tonic
            if next_root not in (root, semitone_down_fifth):
                viol += 1
    if dom == 0:
        return torch.tensor(0.0, device=soprano.device)
    return torch.tensor(viol / dom, dtype=torch.float32, device=soprano.device)

# ------------------------------------------------------------------
# Progression helper: circle‑of‑fifths / stepwise root motion
# ------------------------------------------------------------------
_ALLOWED_ROOT_INTERVALS = {0, 2, 5, 7}  # unison, M2/m2, P4, P5 (mod‑12)
def root_motion_penalty(soprano, alto, tenor, bass, pad_val=PAD_TOKEN):
    """
    Fraction of valid chord transitions where the bass‑defined roots do NOT
    move by an allowed interval (unison = prolong, ±2 semitones stepwise,
    +5 or +7 semitones ≈ ascending 4th / descending 5th circle‑motion).
 
    Lower is better; 1.0 means every transition is illegal.
    """
    batch, seq = soprano.shape
    viol = moves = 0
    for b in range(batch):
        for t in range(seq - 1):
            if soprano[b, t] == pad_val or soprano[b, t+1] == pad_val:
                continue
            root_now  = int(bass[b, t]   % 12)
            root_next = int(bass[b, t+1] % 12)
            interval  = (root_next - root_now) % 12
            moves += 1
            if interval not in _ALLOWED_ROOT_INTERVALS:
                viol += 1
    if moves == 0:
        return torch.tensor(0.0, device=soprano.device)
    return torch.tensor(viol / moves, dtype=torch.float32, device=soprano.device)

def illegal_chord_fraction(soprano, alto, tenor, bass, pad_val=PAD_TOKEN):
    """
    Return scalar tensor = fraction of non‑pad time‑steps whose SATB pitch‑class
    set fails to match ANY allowed structure above (transposition‑invariant).
    """
    batch, seq = soprano.shape
    illegal = valid = 0
    for b in range(batch):
        for t in range(seq):
            if soprano[b, t] == pad_val:
                continue
            pcs = {int(x % 12) for x in
                   (soprano[b, t], alto[b, t], tenor[b, t], bass[b, t])}
            matched = False
            for root in pcs:                      # test every note as potential root
                rel = {(p - root) % 12 for p in pcs}
                if rel in _ALLOWED_CHORD_STRUCTS:
                    matched = True
                    break
            valid += 1
            if not matched:
                illegal += 1
    if valid == 0:
        return torch.tensor(0.0, device=soprano.device)
    return torch.tensor(illegal / valid, dtype=torch.float32, device=soprano.device)

# ------------------------------------------------------------------
# Part‑writing helper: penalise parallel 5ths & 8ves between any voice pair
# ------------------------------------------------------------------
_PARALLEL_INTERVALS = {0, 7}  # perfect unison/octave (0) and fifth (7)
def parallel_motion_penalty(soprano, alto, tenor, bass, pad_val=PAD_TOKEN):
    """
    Returns scalar tensor = proportion of consecutive chord pairs that
    contain *any* parallel 5th or octave between the same two voices.
 
    Simplified rule: check S‑A, S‑T, S‑B, A‑T, A‑B, T‑B.  Parallel if
        1) the interval at time t  is 0 or 7,
        2) the interval at time t+1 is 0 or 7,
        3) both voices actually move.
    """
    voices = [soprano, alto, tenor, bass]  # tensors [batch, seq]
    batch, seq = soprano.shape
    total_pairs = viol_pairs = 0
    for i in range(4):
        for j in range(i + 1, 4):
            v1, v2 = voices[i], voices[j]
            for b in range(batch):
                for t in range(seq - 1):
                    if (v1[b, t] == pad_val or v2[b, t] == pad_val or
                        v1[b, t+1] == pad_val or v2[b, t+1] == pad_val):
                        continue
                    int_now  = abs(int(v1[b, t])   - int(v2[b, t]))   % 12
                    int_next = abs(int(v1[b, t+1]) - int(v2[b, t+1])) % 12
                    if int_now in _PARALLEL_INTERVALS and int_next in _PARALLEL_INTERVALS:
                        if v1[b, t] != v1[b, t+1] and v2[b, t] != v2[b, t+1]:
                            viol_pairs += 1
                    total_pairs += 1
    if total_pairs == 0:
        return torch.tensor(0.0, device=soprano.device)
    return torch.tensor(viol_pairs / total_pairs, dtype=torch.float32, device=soprano.device)

# ------------------------------------------------------------------
# Voice‑crossing penalty: maintain SATB register ordering
# ------------------------------------------------------------------
def voice_crossing_penalty(soprano, alto, tenor, bass, pad_val=PAD_TOKEN):
    """
    Return scalar tensor = fraction of valid chords that violate the
    standard register ordering S ≥ A ≥ T ≥ B.
    """
    batch, seq = soprano.shape
    viol = valid = 0
    for b in range(batch):
        for t in range(seq):
            if soprano[b, t] == pad_val:
                continue
            valid += 1
            if not (soprano[b, t] >= alto[b, t] >= tenor[b, t] >= bass[b, t]):
                viol += 1
    if valid == 0:
        return torch.tensor(0.0, device=soprano.device)
    return torch.tensor(viol / valid, dtype=torch.float32, device=soprano.device)

# ------------------------------------------------------------------
# Large‑leap penalty: discourage leaps larger than a perfect fifth
# ------------------------------------------------------------------
_MAX_CONSONANT_LEAP = 7  # semitones (P5). Change to 12 for octave+
def large_leap_penalty(voice, pad_val=PAD_TOKEN):
    """
    Fraction of melodic intervals > _MAX_CONSONANT_LEAP between successive
    notes in a single voice tensor [batch, seq].
    """
    batch, seq = voice.shape
    viol = moves = 0
    for b in range(batch):
        for t in range(seq - 1):
            if voice[b, t] == pad_val or voice[b, t+1] == pad_val:
                continue
            interval = abs(int(voice[b, t+1]) - int(voice[b, t]))
            moves += 1
            if interval > _MAX_CONSONANT_LEAP:
                viol += 1
    if moves == 0:
        return torch.tensor(0.0, device=voice.device)
    return torch.tensor(viol / moves, dtype=torch.float32, device=voice.device)

##############################
# Training Function with teacher forcing ratio and weighted Bass loss
##############################
def train_model(model, train_loader, optimizer, criterion, num_epochs=10, teacher_forcing_ratio=0.3):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            src = batch['soprano'].to(DEVICE)
            trg = batch['atb'].to(DEVICE)
            optimizer.zero_grad()
            alto_logits, tenor_logits, bass_logits = model(src, trg, tgt_mask=None, teacher_forcing_ratio=teacher_forcing_ratio)
            batch_size, trg_len, vocab_size = alto_logits.shape
            out_alto = alto_logits.reshape(-1, vocab_size)
            out_tenor = tenor_logits.reshape(-1, vocab_size)
            out_bass = bass_logits.reshape(-1, vocab_size)
            trg_alto = trg[:,:,0].reshape(-1)
            trg_tenor = trg[:,:,1].reshape(-1)
            trg_bass = trg[:,:,2].reshape(-1)
            loss_alto = criterion(out_alto, trg_alto)
            loss_tenor = criterion(out_tenor, trg_tenor)
            loss_bass = criterion(out_bass, trg_bass)
            # Soft penalty for vertical dissonances w.r.t. the soprano
            pen_alto  = dissonance_penalty(src, trg[:,:,0])
            pen_tenor = dissonance_penalty(src, trg[:,:,1])
            pen_bass  = dissonance_penalty(src, trg[:,:,2])
            consonance_penalty = (pen_alto + pen_tenor + pen_bass) / 3.0
            # Penalty for chords outside the allowed tonal set
            penalty_illegal = illegal_chord_fraction(src,
                                                     trg[:,:,0], trg[:,:,1], trg[:,:,2])
            # Penalty when V does NOT resolve to I or toggle V↔V7
            penalty_progress = dominant_progress_penalty(src,
                                                         trg[:,:,0], trg[:,:,1], trg[:,:,2])
            # Penalty for root motions outside circle‑of‑fifths / stepwise set
            penalty_root = root_motion_penalty(src,
                                               trg[:,:,0], trg[:,:,1], trg[:,:,2])
            # Penalty for parallel perfect 5ths or octaves
            penalty_parallel = parallel_motion_penalty(src,
                                                       trg[:,:,0], trg[:,:,1], trg[:,:,2])
            # Penalty for voice crossing
            penalty_cross = voice_crossing_penalty(src,
                                                   trg[:,:,0], trg[:,:,1], trg[:,:,2])
            # Penalty for large melodic leaps in each voice
            leap_s = large_leap_penalty(src)
            leap_a = large_leap_penalty(trg[:,:,0])
            leap_t = large_leap_penalty(trg[:,:,1])
            leap_b = large_leap_penalty(trg[:,:,2])
            penalty_leap = (leap_s + leap_a + leap_t + leap_b) / 4.0
            loss = (loss_alto + loss_tenor + 0.3 * loss_bass
                    + 0.2 * consonance_penalty
                    + 0.2 * penalty_illegal
                    + 0.1 * penalty_progress
                    + 0.1 * penalty_root
                    + 0.1 * penalty_parallel
                    + 0.05 * penalty_cross
                    + 0.05 * penalty_leap)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss/len(train_loader):.4f}")

##############################
# Evaluation Function
##############################
def evaluate_model(model, val_loader, criterion):
    model.eval()
    total_loss_alto = total_loss_tenor = total_loss_bass = total_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            src = batch['soprano'].to(DEVICE)
            trg = batch['atb'].to(DEVICE)
            alto_logits, tenor_logits, bass_logits = model(src, trg, tgt_mask=None)
            batch_size, trg_len, vocab_size = alto_logits.shape
            out_alto = alto_logits.reshape(-1, vocab_size)
            out_tenor = tenor_logits.reshape(-1, vocab_size)
            out_bass = bass_logits.reshape(-1, vocab_size)
            trg_alto = trg[:,:,0].reshape(-1)
            trg_tenor = trg[:,:,1].reshape(-1)
            trg_bass = trg[:,:,2].reshape(-1)
            total_loss_alto += criterion(out_alto, trg_alto).item()
            total_loss_tenor += criterion(out_tenor, trg_tenor).item()
            total_loss_bass += criterion(out_bass, trg_bass).item()
            total_batches += 1
    avg_loss_alto = total_loss_alto / total_batches
    avg_loss_tenor = total_loss_tenor / total_batches
    avg_loss_bass = total_loss_bass / total_batches
    print(f"Validation Loss - Alto: {avg_loss_alto:.4f}, Tenor: {avg_loss_tenor:.4f}, Bass: {avg_loss_bass:.4f}")

##############################
# Inference and MIDI Generation Functions
##############################
def generate_predictions(model, test_loader):
    model.eval()
    all_satb = []
    with torch.no_grad():
        for batch in test_loader:
            src = batch['soprano'].to(DEVICE)
            trg = batch['atb'].to(DEVICE)
            trg_len = trg.size(1)
            # Use sample_decode with temperature sampling
            pred = model.sample_decode(src, max_len=trg_len, temperature=1.5, print_debug=False)
            for i in range(src.size(0)):
                satb = pred[i].cpu().tolist()
                soprano_seq = src[i].cpu().tolist()
                combined = []
                for t in range(len(soprano_seq)):
                    if t < len(satb):
                        combined.append([soprano_seq[t]] + satb[t])
                    else:
                        combined.append([soprano_seq[t], SOS_TOKEN, SOS_TOKEN, SOS_TOKEN])
                all_satb.append(combined)
    pred_json_path = os.path.join(PRED_OUTPUT_FOLDER, "satb_predictions.json")
    with open(pred_json_path, "w") as f:
        json.dump(all_satb, f, indent=2)
    print(f"SATB predictions have been saved to '{pred_json_path}'.")

def convert_satb_to_midi(sample_satb, out_midi_path="satb_sample.mid"):
    soprano_part = stream.Part(id="Soprano")
    alto_part = stream.Part(id="Alto")
    tenor_part = stream.Part(id="Tenor")
    bass_part = stream.Part(id="Bass")
    for chord_vals in sample_satb:
        s_note = note.Note()
        s_note.pitch.midi = chord_vals[0]
        s_note.quarterLength = 1
        a_note = note.Note()
        a_note.pitch.midi = chord_vals[1]
        a_note.quarterLength = 1
        t_note = note.Note()
        t_note.pitch.midi = chord_vals[2]
        t_note.quarterLength = 1
        b_note = note.Note()
        b_note.pitch.midi = chord_vals[3]
        b_note.quarterLength = 1
        soprano_part.append(s_note)
        alto_part.append(a_note)
        tenor_part.append(t_note)
        bass_part.append(b_note)
    score = stream.Score([soprano_part, alto_part, tenor_part, bass_part])
    score.write("midi", fp=out_midi_path)
    print(f"MIDI file '{out_midi_path}' has been created.")
    return score

##############################
# Dataset and DataLoader Definitions
##############################
class ChoraleDataset(Dataset):
    def __init__(self, input_folder, output_folder):
        self.input_files = sorted([os.path.join(input_folder, f)
                                   for f in os.listdir(input_folder) if f.endswith('.json')])
        self.output_files = sorted([os.path.join(output_folder, f)
                                    for f in os.listdir(output_folder) if f.endswith('.json')])
        assert len(self.input_files) == len(self.output_files), "Mismatch in number of input and output files."
    def __len__(self):
        return len(self.input_files)
    def __getitem__(self, idx):
        with open(self.input_files[idx], 'r') as f:
            input_seq = json.load(f)
        with open(self.output_files[idx], 'r') as f:
            output_seq = json.load(f)
        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        if AUGMENT_TRANSPOSITION:
            shift = random.randint(*TRANSP_RANGE)
            input_seq  = [(p + shift) % 128 for p in input_seq]
            output_seq = [[(n + shift) % 128 for n in chord]
                          for chord in output_seq]
        output_tensor = torch.tensor(output_seq, dtype=torch.long)
        return {'soprano': input_tensor, 'atb': output_tensor}

def collate_fn(batch):
    inputs = [sample['soprano'] for sample in batch]
    outputs = [sample['atb'] for sample in batch]
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=PAD_TOKEN)
    padded_outputs = pad_sequence(outputs, batch_first=True, padding_value=PAD_TOKEN)
    return {'soprano': padded_inputs, 'atb': padded_outputs}

def create_dataloaders(dataset, batch_size=1, test_ratio=0.05, val_ratio=0.20):
    total_size = len(dataset)
    test_size = int(test_ratio * total_size)
    val_size = int(val_ratio * total_size)
    train_size = total_size - val_size - test_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Dataset split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)} samples.")
    return train_loader, val_loader, test_loader

##############################
# Main Execution
##############################
if __name__ == "__main__":
    # Create dataset and dataloaders.
    dataset = ChoraleDataset(INPUT_FOLDER, OUTPUT_FOLDER)
    print(f"Total samples (paragraphs): {len(dataset)}")
    train_loader, val_loader, test_loader = create_dataloaders(dataset, batch_size=1)
    
    # Build the Transformer-based model.
    model, optimizer, criterion = build_model()
    
    # Train the model with low teacher forcing ratio.
    EPOCHS_RUN = 3   # 2‑4 epochs are usually good enough
    print("Starting training...")
    train_model(model, train_loader, optimizer, criterion,
                num_epochs=EPOCHS_RUN, teacher_forcing_ratio=0.3)

    # Save trained model checkpoint
    ckpt_path = os.path.join(MODEL_OUTPUT_FOLDER,
                             f"transformer_harmonizer_epoch{EPOCHS_RUN}.pt")
    save_checkpoint(model, optimizer, epoch=EPOCHS_RUN, path=ckpt_path)
    
    # Evaluate the model.
    evaluate_model(model, val_loader, criterion)
    
    # Generate predictions on the test set using temperature sampling.
    generate_predictions(model, test_loader)
    
    # Convert the first SATB prediction to MIDI.
    pred_json_path = os.path.join(PRED_OUTPUT_FOLDER, "satb_predictions.json")
    with open(pred_json_path, "r") as f:
        satb_data = json.load(f)
    if len(satb_data) > 0:
        sample_satb = satb_data[0]
        midi_path = os.path.join(PRED_OUTPUT_FOLDER, "satb_sample.mid")
        score = convert_satb_to_midi(sample_satb, out_midi_path=midi_path)
        pm = pretty_midi.PrettyMIDI(midi_path)
        piano_roll = pm.get_piano_roll(fs=100)
        plt.figure(figsize=(12,6))
        plt.imshow(piano_roll, aspect="auto", origin="lower", cmap="gray_r")
        plt.xlabel("Time Frames")
        plt.ylabel("MIDI Pitch")
        plt.title(f"Piano Roll of {os.path.basename(midi_path)}")
        plt.colorbar(label="Velocity")
        plt.show()
    else:
        print("No SATB predictions found for MIDI conversion.")