from pathlib import Path
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class UnitSequenceDataset(Dataset):
    """
    Dataset for HuBERT unit sequences stored as .npy files.

    Expected directory structure:

        base_dir/
            train/
                .../*.npy
            valid/
                .../*.npy
            test/
                .../*.npy
    """

    def __init__(
        self,
        base_dir,
        split="train",
        seq_len=None,
        pad_value=-100,
        random_crop=True,
        file_ext=".npy",
    ):
        self.base_dir = Path(base_dir)
        self.split = split
        self.seq_len = seq_len
        self.pad_value = pad_value
        self.random_crop = random_crop
        self.file_ext = file_ext

        split_dir = self.base_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        pattern = str(split_dir / f"**/*{file_ext}")
        self.files = [Path(p) for p in glob(pattern, recursive=True)]
        self.files.sort()

        if len(self.files) == 0:
            raise RuntimeError(f"No '{file_ext}' files found in {split_dir}")

    def __len__(self):
        return len(self.files)

    def _crop_or_pad(self, units):
        original_len = units.size(0)

        if self.seq_len is None:
            return units, original_len

        if original_len == self.seq_len:
            return units, original_len

        if original_len > self.seq_len:
            # crop
            if self.random_crop:
                max_start = original_len - self.seq_len
                start = torch.randint(0, max_start + 1, (1,)).item()
            else:
                start = max((original_len - self.seq_len) // 2, 0)
            units = units[start:start + self.seq_len]
            return units, original_len

        # pad
        pad_len = self.seq_len - original_len
        pad = torch.full((pad_len,), self.pad_value, dtype=units.dtype)
        units = torch.cat([units, pad], dim=0)
        return units, original_len

    def __getitem__(self, idx):
        path = self.files[idx]
        arr = np.load(path)
        units = torch.from_numpy(arr).long()

        units, original_len = self._crop_or_pad(units)

        return {
            "units": units,
            "length": original_len,
            "path": str(path),
        }


def unit_collate_fn(batch, pad_value=-100):
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    seqs = [b["units"] for b in batch]
    max_len = max(s.size(0) for s in seqs)

    if all(s.size(0) == max_len for s in seqs):
        units = torch.stack(seqs, dim=0)
    else:
        padded = []
        for s in seqs:
            if s.size(0) < max_len:
                pad_len = max_len - s.size(0)
                pad = torch.full((pad_len,), pad_value, dtype=s.dtype)
                s = torch.cat([s, pad], dim=0)
            padded.append(s)
        units = torch.stack(padded, dim=0)

    paths = [b["path"] for b in batch]

    return {
        "units": units,
        "lengths": lengths,
        "paths": paths,
    }


class UnitDataModule(pl.LightningDataModule):
    """
    LightningDataModule for HuBERT unit sequences.
    """

    def __init__(
        self,
        base_dir,
        batch_size=32,
        num_workers=4,
        seq_len=None,
        pad_value=-100,
        random_crop=True,
        train_split="train",
        val_split="valid",
        test_split="test",
        pin_memory=True,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_len = seq_len
        self.pad_value = pad_value
        self.random_crop = random_crop
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.pin_memory = pin_memory

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self._train_dataset = UnitSequenceDataset(
                base_dir=self.base_dir,
                split=self.train_split,
                seq_len=self.seq_len,
                pad_value=self.pad_value,
                random_crop=self.random_crop,
            )
            self._val_dataset = UnitSequenceDataset(
                base_dir=self.base_dir,
                split=self.val_split,
                seq_len=self.seq_len,
                pad_value=self.pad_value,
                random_crop=False,
            )

        if stage in (None, "test"):
            self._test_dataset = UnitSequenceDataset(
                base_dir=self.base_dir,
                split=self.test_split,
                seq_len=self.seq_len,
                pad_value=self.pad_value,
                random_crop=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            collate_fn=lambda b: unit_collate_fn(b, pad_value=self.pad_value),
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            collate_fn=lambda b: unit_collate_fn(b, pad_value=self.pad_value),
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            collate_fn=lambda b: unit_collate_fn(b, pad_value=self.pad_value),
        )