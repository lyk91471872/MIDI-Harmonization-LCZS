from datasets import Dataset
from pyvoicing import Voicing, Rest
from os import listdir
from os.path import isdir, join
import mido

RAW_DIR = 'cocochorales'
OUTPUT_DIR = 'dataset_coco_abc'

def get_chorales() -> tuple[str]:
    chorales = []
    for split in ('train', 'valid', 'test'):
        if isdir(split_dir:=join(RAW_DIR, split)):
            for chorale in sorted(listdir(split_dir)):
                if isdir(chorale_dir:=join(split_dir, chorale)):
                    chorales.append((split, chorale, chorale_dir))
    return chorales

def format_abc(s, a, t, b):
    return '\n'.join([f'[V:S] {s}', f'[V:A] {a}', f'[V:T] {t}', f'[V:B] {b}'])

def parse_midi(midi_events) -> list[int]:
    # return note values, Rest() for rest

def main(coco_dir='cocochorales', out_dir='dataset_coco_abc'):
    data = []
    for split, chorale, chorale_dir in get_chorales():
        tracks = mido.MidiFile(join(chorale_dir, 'mix.mid')).tracks[1:]
        voices = [parse_midi(_) for _ in tracks]
        # TODO
        for semitone in KEY_TRANS:
            transposed = [v.transpose(semitone) for v in voicings]
            abc_lines = [v.abc.split() for v in transposed]
            soprano, alto, tenor, bass = zip(*abc_lines)
            s = ' '.join(soprano)
            a = ' '.join(alto)
            t = ' '.join(tenor)
            b = ' '.join(bass)
            data.append({
                'id': f'{split}_{track}_trans{semitone}',
                'instruction': 'Harmonize the below soprano melody into SATB form.',
                'input': s,
                'output': format_abc(s, a, t, b)
            })
    ds = Dataset.from_list(data)
    ds.save_to_disk(out_dir)
    print(f'Saved {len(ds)} examples to {out_dir}')


if __name__ == '__main__':
    main()
