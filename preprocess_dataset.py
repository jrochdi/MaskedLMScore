#!/usr/bin/env python3
"""
Build an LRS3 HuBERT-unit dataset from an existing LRS3 installation.

Original structure (example):
    /data/datasets/LRS3/
        pretrain/...
        trainval/emn28FrJ6CI/50001.wav, 50001.txt, 50001.mp4, ...
        test/...

New structure (example):
    <WORKSPACE_ROOT>/dataset/
        pretrain/...
        trainval/emn28FrJ6CI/50001.npy, 50001.txt, ...
        test/...

- mp4 files are NOT copied.
- wav -> npy (HuBERT unit sequences).
- One workspace folder holds:
    - saved HuBERT model checkpoint
    - trained k-means
    - new dataset
"""

# ==============================
# CONFIG (EDIT THESE)
# ==============================
from pathlib import Path

# Path to original LRS3 dataset
LRS3_ROOT = Path("/data/datasets/LRS3")

# Workspace root (this script will create this folder if it doesn't exist).
WORKSPACE_ROOT = Path("LRS3_HUBERT_WORKSPACE")
DATASET_NAME = "lrs3-gibberish-hubert"

# Name of the HuBERT bundle (torchaudio.pipelines)
HUBERT_BUNDLE_NAME = "HUBERT_BASE"

# Which HuBERT layer to use
HUBERT_LAYER = 6

# KMeans settings
N_CLUSTERS = 100
MAX_FRAMES_FOR_KMEANS = 2_000_000
KMEANS_BATCH_SIZE_FRAMES = 10_000

# splits used for k-means training
KMEANS_SPLITS = ["pretrain", "trainval"]

# Device
DEVICE = "cuda"

# random seed
RANDOM_SEED = 1337

# resample audio to HuBERT expected rate
TARGET_SAMPLE_RATE = 16000

# ==============================
# SCRIPT START
# ==============================
import os
import random
import shutil
import numpy as np
import torch
import torchaudio
from torchaudio import pipelines
from sklearn.cluster import MiniBatchKMeans
import joblib
from tqdm import tqdm

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Derived paths
MODEL_DIR = WORKSPACE_ROOT / "hubert_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
HUBERT_CHECKPOINT_PATH = MODEL_DIR / "hubert_base_state_dict.pt"

KMEANS_PATH = WORKSPACE_ROOT / f"kmeans_hubert_layer{HUBERT_LAYER}_k{N_CLUSTERS}.joblib"

DATASET_OUT_ROOT = WORKSPACE_ROOT / DATASET_NAME
DATASET_OUT_ROOT.mkdir(parents=True, exist_ok=True)


def get_hubert_model():
    print(f"Loading HuBERT model (bundle: {HUBERT_BUNDLE_NAME})...")
    bundle = getattr(pipelines, HUBERT_BUNDLE_NAME)

    if HUBERT_CHECKPOINT_PATH.exists():
        print(f"  Found checkpoint at {HUBERT_CHECKPOINT_PATH}, loading...")
        model = bundle.get_model()
        state_dict = torch.load(HUBERT_CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        print("  No checkpoint found — downloading model and saving...")
        model = bundle.get_model()
        torch.save(model.state_dict(), HUBERT_CHECKPOINT_PATH)
        print(f"  Saved checkpoint to {HUBERT_CHECKPOINT_PATH}")

    model.to(DEVICE)
    model.eval()
    print(f"  Loaded HuBERT model on {DEVICE}.")
    return model, bundle.sample_rate


def get_resampler(orig_sr, target_sr):
    if orig_sr == target_sr:
        return None
    return torchaudio.transforms.Resample(orig_sr, target_sr)


def load_and_prepare_waveform(path, target_sr):
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = get_resampler(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    return waveform, sr


def extract_layer_features(model, waveform, layer_idx):
    waveform = waveform.to(DEVICE)
    with torch.no_grad():
        feats_list, _ = model.extract_features(waveform, num_layers=layer_idx)
    return feats_list[-1].squeeze(0).cpu()


def iter_lrs3_wavs(splits):
    for split in splits:
        split_root = LRS3_ROOT / split
        if not split_root.exists():
            print(f"WARNING: split root {split_root} missing, skipping.")
            continue
        for dirpath, _, filenames in os.walk(split_root):
            for fn in filenames:
                if fn.lower().endswith(".wav"):
                    yield split, Path(dirpath) / fn


def collect_features_for_kmeans(model, target_sr):
    print("Collecting features for k-means training...")

    all_wavs = list(iter_lrs3_wavs(KMEANS_SPLITS))
    random.shuffle(all_wavs)  # shuffle in-place
    
    frames_collected = 0
    feats_buffer = []

    for split, wav_path in tqdm(all_wavs, desc="Extracting HuBERT frames"):
        if frames_collected >= MAX_FRAMES_FOR_KMEANS:
            break

        waveform, sr = load_and_prepare_waveform(wav_path, target_sr)
        feats = extract_layer_features(model, waveform, HUBERT_LAYER)
        n_frames = feats.shape[0]
        remaining = MAX_FRAMES_FOR_KMEANS - frames_collected

        if n_frames <= remaining:
            chosen = feats
        else:
            idx = np.random.choice(n_frames, size=remaining, replace=False)
            chosen = feats[idx]

        feats_buffer.append(chosen.numpy())
        frames_collected += chosen.shape[0]

    if not feats_buffer:
        raise RuntimeError("No features collected for k-means.")

    all_feats = np.concatenate(feats_buffer, axis=0)
    print(f"  Total collected frames: {all_feats.shape[0]}")
    return all_feats


def train_kmeans(features):
    print("Training MiniBatchKMeans...")
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=KMEANS_BATCH_SIZE_FRAMES,
        random_state=RANDOM_SEED,
        verbose=1,
    )
    kmeans.fit(features)
    joblib.dump(kmeans, KMEANS_PATH)
    print(f"  Saved k-means model to {KMEANS_PATH}")
    return kmeans


def load_or_train_kmeans(model, target_sr):
    if KMEANS_PATH.exists():
        print(f"Found existing k-means at {KMEANS_PATH}, loading...")
        return joblib.load(KMEANS_PATH)

    feats = collect_features_for_kmeans(model, target_sr)
    kmeans = train_kmeans(feats)
    del feats
    return kmeans


def rebuild_dataset_with_units(model, kmeans, target_sr):
    print("Rebuilding dataset with HuBERT unit sequences...")
    splits = ["pretrain", "trainval", "test"]

    for split in splits:
        in_root = LRS3_ROOT / split
        if not in_root.exists():
            print(f"WARNING: split {split} missing, skipping.")
            continue

        out_root = DATASET_OUT_ROOT / split
        wav_paths = []

        for dirpath, _, filenames in os.walk(in_root):
            for fn in filenames:
                if fn.lower().endswith(".wav"):
                    wav_paths.append(Path(dirpath) / fn)

        for wav_path in tqdm(wav_paths, desc=f"Processing {split}"):
            rel = wav_path.parent.relative_to(in_root)
            out_dir = out_root / rel
            out_dir.mkdir(parents=True, exist_ok=True)

            # copy txt files in same directory
            for txt in wav_path.parent.glob("*.txt"):
                out_txt = out_dir / txt.name
                if not out_txt.exists():
                    shutil.copy2(txt, out_txt)

            # process wav → npy
            base = wav_path.stem
            out_npy = out_dir / f"{base}.npy"
            if out_npy.exists():
                continue

            waveform, sr = load_and_prepare_waveform(wav_path, target_sr)
            feats = extract_layer_features(model, waveform, HUBERT_LAYER)
            units = kmeans.predict(feats.numpy().astype(np.float32))
            np.save(out_npy, units.astype(np.int32))

    print("Done. Output dataset is in:")
    print(DATASET_OUT_ROOT)


def main():
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

    model, hubert_sr = get_hubert_model()

    if hubert_sr != TARGET_SAMPLE_RATE:
        print(f"WARNING: HuBERT model expects {hubert_sr}, TARGET_SAMPLE_RATE={TARGET_SAMPLE_RATE}")

    kmeans = load_or_train_kmeans(model, TARGET_SAMPLE_RATE)

    rebuild_dataset_with_units(model, kmeans, TARGET_SAMPLE_RATE)


if __name__ == "__main__":
    main()