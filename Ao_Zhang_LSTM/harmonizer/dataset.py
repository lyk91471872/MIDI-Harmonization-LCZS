import torch
from torch.utils.data import Dataset
import numpy as np

class HarmonyDataset(Dataset):
    def __init__(self, filepath):
        # 假设你已经把数据预处理成 npz 格式，包含:
        # melody: (N, T, 3)  # pitch,duration,beat_pos
        # alto, tenor, bass: (N, T)
        data = np.load(filepath)
        self.melody = data['melody']      # float32
        self.alto = data['alto']          # int64
        self.tenor = data['tenor']
        self.bass = data['bass']

    def __len__(self):
        return len(self.melody)

    def __getitem__(self, idx):
        # 返回 tensor
        m = torch.from_numpy(self.melody[idx])
        a = torch.from_numpy(self.alto[idx])
        t = torch.from_numpy(self.tenor[idx])
        b = torch.from_numpy(self.bass[idx])
        return m, a, t, b
