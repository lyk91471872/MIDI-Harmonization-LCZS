import pickle
import random
from datasets import Dataset, DatasetDict
from numbers import Integral
from pyvoicing import *
from tqdm import tqdm

def to_satb(values: tuple[Integral]) -> Voicing:
    if len(values) == 0:
        return Voicing([Rest()]*4)
    v = Voicing(int(_) for _ in values)
    if (missing:=4-len(v)):
        v += v[-1] * missing
    return v

def to_var_len_samples(voicings: list[Voicing], k: int) -> list[list[Voicing]]:
    assert k >= 2, "k must be at least 2"
    n = len(voicings)
    step = n // k
    lengths = [step*i for i in range(1, k)]
    samples = [voicings]
    for l in lengths:
        i = random.randint(0, n-l)
        samples.append(voicings[i:i+l])
    return samples

def to_12_keys(voicings: list[Voicing]) -> list[list[Voicing]]:
    return [[v>>i for v in voicings] for i in range(-6, 6)]

def main():
    with open('jsb-chorales-quarter.pkl', 'rb') as f:
        data_splits = pickle.load(f)
    ds_splits = dict()
    for split, chorales in data_splits.items():
        data_dicts = []
        for i, chorale in tqdm(enumerate(chorales)):
            voicings = [to_satb(_) for _ in chorale]    # replicate s to full satb
            samples = to_var_len_samples(voicings, 5)
            transpositions = [_ for s in samples for _ in to_12_keys(s)]
            for t in transpositions:
                satb = [_.int_list for _ in t]
                s = [_[-1] for _ in satb]
                data_dicts.append({
                    'id': i,
                    'instruction': 'Harmonize the below soprano melody into SATB form.',
                    'input': str(s),
                    'output': str(satb)
                })
        ds_splits[split] = Dataset.from_list(data_dicts)
    DatasetDict(ds_splits).save_to_disk('dataset_jsb')

if __name__ == '__main__':
    main()
