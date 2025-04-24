import pickle
from datasets import Dataset
from numbers import Integral
from pyvoicing import *

def to_satb(values: tuple[Integral]) -> Voicing:
    if len(values) == 0:
        return Voicing([Rest()]*4)
    v = Voicing(int(_) for _ in values)
    if (missing:=4-len(v)):
        v += v[-1] * missing
    return v

def format(s, a, t, b) -> str:
    return '\n'.join((
        f'[V:S] {s}',
        f'[V:A] {a}',
        f'[V:T] {t}',
        f'[V:B] {b}',
    ))

def main():
    with open('jsb-chorales-quarter.pkl', 'rb') as f:
        data_splits = pickle.load(f)
    data_dicts = []
    for split, chorales in data_splits.items():
        for i, chorale in enumerate(chorales):
            voicings = [to_satb(_) for _ in chorale]
            abc_12keys = [[(v>>i).abc for v in voicings] for i in range(-6, 6)]
            for j, abc in enumerate(abc_12keys):
                b, t, a, s = (' '.join(_) for _ in zip(*abc))
                data_dicts.append({
                    'id': f'{split}{i}_trans{j-6}',
                    'instruction': 'Harmonize the below soprano melody into SATB form.',
                    'input': s,
                    'output': format(s, a, t, b)
                })
    dataset = Dataset.from_list(data_dicts)
    dataset.save_to_disk('dataset_jsb_abc')

if __name__ == '__main__':
    main()
