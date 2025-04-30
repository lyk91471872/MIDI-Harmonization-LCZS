from pyvoicing import *

def parse_abc(abc: str):
    """Assuming legal SATB format"""
    voices = map(lambda _: _.split(' ')[1:], abc.split('\n'))
    n = max(len(_) for _ in voices)
    for v in voices:
        v += ['z'] * (n-len(v))
    voicings = map(Voicing, list(zip(*voices))
