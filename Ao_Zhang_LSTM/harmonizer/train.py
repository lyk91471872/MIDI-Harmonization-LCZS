#!/usr/bin/env python3
# train.py with progress bars and OneCycleLR

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# 导入你的 LSTM 模型定义和常量
from model import LSTMHarmonizer, EMBED_DIM, LSTM_HIDDEN, VOCAB_SIZE, NUM_VOICES

class CocoTinyDataset(Dataset):
    """
    加载 *_coco_tiny.npz，过滤掉长度为 0 的样本。
    输入: mel-roll (T,128) → FloatTensor
    目标: Alto/Tenor/Bass pitch-index (3, T) → LongTensor
    """
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        raw_mels   = data['mel']
        raw_altos  = data['alto']
        raw_tenors = data['tenor']
        raw_basses = data['bass']

        self.samples = []
        for mel, alto, tenor, bass in zip(raw_mels, raw_altos, raw_tenors, raw_basses):
            if mel.shape[0] > 0:
                self.samples.append((mel, alto, tenor, bass))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mel, alto, tenor, bass = self.samples[idx]
        mel_tensor = torch.from_numpy(mel.astype(np.float32))  # (T,128)

        def roll2idx(roll: np.ndarray) -> torch.Tensor:
            idxs = np.argmax(roll, axis=1).astype(np.int64)
            return torch.from_numpy(idxs)

        alto_idx  = roll2idx(alto)
        tenor_idx = roll2idx(tenor)
        bass_idx  = roll2idx(bass)
        target    = torch.stack([alto_idx, tenor_idx, bass_idx], dim=0)  # (3, T)
        return mel_tensor, target


def train(args):
    # 路径配置
    train_npz = os.path.join(args.data_dir, 'train_coco_tiny.npz')
    valid_npz = os.path.join(args.data_dir, 'valid_coco_tiny.npz')

    # 数据集与加载器
    train_ds = CocoTinyDataset(train_npz)
    valid_ds = CocoTinyDataset(valid_npz)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    # 设备 与 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMHarmonizer(input_dim=EMBED_DIM,
                           hidden_dim=LSTM_HIDDEN,
                           output_dim=VOCAB_SIZE).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # OneCycleLR 调度（每个 batch 调用 scheduler.step()）
    total_steps = args.num_epochs * len(train_loader)
    scheduler = OneCycleLR(optimizer,
                           max_lr=args.lr,
                           total_steps=total_steps,
                           pct_start=0.3,
                           anneal_strategy='cos',
                           cycle_momentum=False)

    # 恢复 checkpoint
    start_epoch = args.start_epoch
    best_val_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint '{args.resume}' and resuming from epoch {start_epoch}")
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state)

    step = 0
    # 训练循环
    for epoch in range(start_epoch, args.num_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch:02d} Train", unit="batch") as train_bar:
            for mel, target in train_bar:
                mel    = mel.to(device)
                target = target.squeeze(0).to(device)

                optimizer.zero_grad()
                outputs = model(mel)  # list of tensors (batch,T,VOCAB_SIZE)

                loss = 0.0
                for i, head_out in enumerate(outputs):
                    logits = head_out.view(-1, VOCAB_SIZE)
                    gt     = target[i]
                    loss  += criterion(logits, gt)
                loss = loss / NUM_VOICES
                loss.backward()
                optimizer.step()

                # step-level 调度
                scheduler.step()
                step += 1

                train_loss += loss.item()
                avg_train = train_loss / step
                lr = scheduler.get_last_lr()[0]
                train_bar.set_postfix({'avg_loss': f"{avg_train:.4f}", 'lr': f"{lr:.6f}"})

        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        with tqdm(valid_loader, desc=f"Epoch {epoch:02d} Valid", unit="batch") as valid_bar:
            for mel, target in valid_bar:
                mel    = mel.to(device)
                target = target.squeeze(0).to(device)
                outputs = model(mel)
                batch_l = 0.0
                for i, head_out in enumerate(outputs):
                    logits = head_out.view(-1, VOCAB_SIZE)
                    gt     = target[i]
                    batch_l += criterion(logits, gt)
                batch_l = batch_l / NUM_VOICES
                val_loss += batch_l.item()
                val_count += 1
                avg_val = val_loss / val_count
                valid_bar.set_postfix({'avg_loss': f"{avg_val:.4f}"})

        avg_val = val_loss / val_count
        print(f"Epoch {epoch:02d} | Valid Loss: {avg_val:.4f}")

        # 保存 checkpoint
        ckpt_epoch = f"lstm_epoch{epoch:02d}.pth"
        torch.save(model.state_dict(), ckpt_epoch)
        print(f"Checkpoint saved: {ckpt_epoch}")

        # 保存最佳模型
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'lstm_best.pth')
            print(f"New best model saved at epoch {epoch:02d}, val_loss={avg_val:.4f}")

    print("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM Harmonizer with OneCycleLR and progress bars')
    parser.add_argument('--data-dir',    type=str,
                        default='/Volumes/Kid_Rage 1/[_Database_]/CocoChorales/cocochorales_tiny_v1_midi')
    parser.add_argument('--batch-size',  type=int,   default=1)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--num-epochs',  type=int,   default=20)
    parser.add_argument('--resume',      type=str,   default='',    help='checkpoint to resume from')
    parser.add_argument('--start-epoch', type=int,   default=1,     help='epoch to start from')
    args = parser.parse_args()
    train(args)
