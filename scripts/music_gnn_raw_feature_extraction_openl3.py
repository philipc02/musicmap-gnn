#!/usr/bin/env python
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf  # ensures mp3 decoding on some platforms
import torch
import torchopenl3

def extract_openl3_vector(
    file_path: str,
    sr: int = 48000,
    input_repr: str = "mel256",
    content_type: str = "music",
    embedding_size: int = 512,
    hop_size: float = 0.5,
    batch_size: int = 32,
    device: str = None,
):
    """
    Returns a (D,) mean-pooled OpenL3 embedding for a single track.
    """
    try:
        # Load mono and resample to 48k for OpenL3
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        if y.size == 0:
            return None

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # (T, D) embeddings over time windows
        emb, _ = torchopenl3.get_audio_embedding(
            y,
            sr,
            input_repr=input_repr,
            content_type=content_type,
            embedding_size=embedding_size,
            center=True,
            hop_size=hop_size,
            batch_size=batch_size,
            device=device,
        )

        if emb is None or len(emb) == 0:
            return None

        # Mean-pool over time -> (D,)
        vec = emb.mean(axis=0)
        return vec.astype(np.float32)

    except Exception as e:
        print(f"[WARN] Failed on {file_path}: {e}")
        return None
    

def list_mp3s(audio_dir: str):
    for root, _, files in os.walk(audio_dir):
        for f in files:
            if f.lower().endswith(".mp3"):
                yield os.path.join(root, f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True, help="Path to fma_small directory")
    ap.add_argument("--out_parquet", default="../data/fma_small_openl3.parquet",
                    help="Output Parquet file (features)")
    ap.add_argument("--out_numpy", default="../data/fma_small_openl3.npy",
                    help="Optional NumPy .npy dump of the feature matrix")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--input_repr", choices=["mel256", "mel128", "linear"], default="mel256")
    ap.add_argument("--content_type", choices=["music", "env"], default="music")
    ap.add_argument("--embedding_size", type=int, choices=[512, 6144], default=512)
    ap.add_argument("--hop_size", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    paths = list(list_mp3s(args.audio_dir))
    print(f"Found {len(paths)} MP3 files under {args.audio_dir}")

    features = []
    idxs = []
    skipped = 0

    for p in tqdm(paths, desc="Extracting OpenL3"):
        # FMA filenames are zero-padded track IDs like 000123.mp3
        basename = os.path.splitext(os.path.basename(p))[0]
        try:
            track_id = int(basename)  # keep as int to match tracks.csv index
        except ValueError:
            skipped += 1
            continue

        vec = extract_openl3_vector(
            p,
            sr=args.sr,
            input_repr=args.input_repr,
            content_type=args.content_type,
            embedding_size=args.embedding_size,
            hop_size=args.hop_size,
            batch_size=args.batch_size,
        )
        if vec is None:
            skipped += 1
            continue

        idxs.append(track_id)
        features.append(vec)

    if not features:
        raise RuntimeError("No features extracted — check paths/codecs/dependencies.")

    X = np.vstack(features)  # shape (N, D)
    df = pd.DataFrame(X, index=pd.Index(idxs, name="track_id"))
    df.to_parquet(args.out_parquet, index=True)
    np.save(args.out_numpy, X)

    print(f"Saved {df.shape} features to {args.out_parquet}")
    print(f"Also wrote matrix to {args.out_numpy}")
    print(f"Skipped files: {skipped}")

if __name__ == "__main__":
    main()