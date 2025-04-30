#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import pretty_midi
from tqdm import tqdm

# ====== 配置 ======
DATA_ROOT = Path('/Volumes/Kid_Rage 1/[_Database_]/CocoChorales/cocochorales_tiny_v1_midi')
SPLITS    = ['train', 'valid', 'test']
FS        = 8   # 每四分音符帧数
OUT_DIR   = DATA_ROOT

# 根据文件名判声部
def get_role(fname: str) -> str:
    n = fname.lower()
    if n.startswith('1_') or 'trumpet' in n:    return 'mel'
    if n.startswith('2_') or 'horn' in n or 'alto' in n:   return 'alto'
    if n.startswith('3_') or 'trombone' in n or 'tenor' in n: return 'tenor'
    if n.startswith('4_') or 'tuba' in n or 'bass' in n:     return 'bass'
    return None

# 单文件转 roll
def midi_to_roll(path: Path) -> np.ndarray:
    pm = pretty_midi.PrettyMIDI(str(path))
    pr = pm.get_piano_roll(fs=FS)           # (128, frames)
    return (pr>0).astype(np.uint8).T        # -> (frames,128)

def process_split(split: str):
    split_dir = DATA_ROOT / split
    if not split_dir.exists():
        print(f"[Warn] {split_dir} 不存在，跳过")
        return [],[],[],[]

    # 临时按 track_id 收集： track_id -> role -> list of rolls
    grouped = {}
    # 遍历所有子文件夹
    for track_dir in tqdm(sorted(split_dir.iterdir()), desc=split):
        if not track_dir.is_dir(): continue
        # 递归查 .mid
        for path in track_dir.rglob('*.mid'):
            role = get_role(path.name)
            if not role: continue
            try:
                roll = midi_to_roll(path)
            except Exception:
                continue
            if roll.shape[0]==0: continue
            grouped.setdefault(track_dir.name, {}).setdefault(role, []).append(roll)

    mels, altos, tenors, basses = [],[],[],[]
    for track_id, roles in grouped.items():
        # 至少需要 melody
        if 'mel' not in roles: continue
        # 合并每个 role 下的多轨：按位 OR
        merged = {}
        maxlen = 0
        for r in ['mel','alto','tenor','bass']:
            arrs = roles.get(r, [])
            if not arrs:
                merged[r] = np.zeros((0,128),dtype=np.uint8)
            else:
                L = max(a.shape[0] for a in arrs)
                M = np.zeros((L,128),dtype=np.uint8)
                for a in arrs:
                    pad = np.zeros((L - a.shape[0], 128),dtype=np.uint8)
                    M |= np.vstack([a, pad]) if a.shape[0]<L else a
                merged[r] = M
            maxlen = max(maxlen, merged[r].shape[0])
        if maxlen==0: continue
        # 再次 pad（可选，多数都一致）
        for r in merged:
            a = merged[r]
            if a.shape[0]<maxlen:
                merged[r] = np.vstack([a, np.zeros((maxlen-a.shape[0],128),dtype=np.uint8)])
        mels.append(merged['mel'])
        altos.append(merged['alto'])
        tenors.append(merged['tenor'])
        basses.append(merged['bass'])

    print(f"{split}: 有效 track 数 = {len(mels)}")
    return mels, altos, tenors, basses

def main():
    for split in SPLITS:
        mels, altos, tenors, basses = process_split(split)
        out = OUT_DIR / f"{split}_coco_tiny.npz"
        np.savez_compressed(str(out),
                            mel   = np.array(mels,   dtype=object),
                            alto  = np.array(altos,  dtype=object),
                            tenor = np.array(tenors, dtype=object),
                            bass  = np.array(basses, dtype=object))
        mb = out.stat().st_size/1024**2
        print(f"Saved {out.name} | samples={len(mels)} | size={mb:.2f} MB")

if __name__=='__main__':
    main()
