from pyvoicing import Voicing

def parse_abc(abc: str):
    """Assuming legal SATB format"""
    voices = [_.split(' ')[1:] for _ in abc.strip().split('\n')]
    n = max(len(_) for _ in voices)
    for v in voices:
        v += ['z'] * (n-len(v))
    voicings = map(Voicing, list(zip(*voices)))
    return list(voicings)
