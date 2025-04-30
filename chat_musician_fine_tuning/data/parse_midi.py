from mido import MidiFile
from pyvoicing import Rest, Voicing
from typing import List

def midi_to_satb(midi_path: str, sub_divisions: int = 4) -> List[Voicing]:
    """
    Load a MIDI file and convert tracks 1–4 into SATB voicings.
    Rests use Rest(), chords are Voicing objects.
    """
    mid = MidiFile(midi_path)
    tpq = mid.ticks_per_beat
    res = tpq // sub_divisions

    # Collect note events per SATB track
    events = []
    for track in mid.tracks[1:5]:
        t = 0
        ev = []
        for msg in track:
            t += msg.time
            if msg.type == 'note_on':
                ev.append((t, msg.note, msg.velocity > 0))
            elif msg.type == 'note_off':
                ev.append((t, msg.note, False))
        events.append(sorted(ev, key=lambda e: e[0]))

    # Determine steps and build sequences
    last = max(ev[-1][0] for ev in events)
    steps = last // res + 1
    sequences = []

    for ev in events:
        seq, idx, cur = [], 0, Rest()
        for i in range(steps):
            tick = i * res
            while idx < len(ev) and ev[idx][0] <= tick:
                _, note, on = ev[idx]
                cur = note if on else Rest()
                idx += 1
            seq.append(cur)
        sequences.append(seq)

    # Zip into Voicing list
    return [Voicing(notes) for notes in zip(*sequences)]

# Example usage
if __name__ == '__main__':
    chords = midi_to_satb('mix.mid', 4)
    print(f"Generated {len(chords)} voicings.")
