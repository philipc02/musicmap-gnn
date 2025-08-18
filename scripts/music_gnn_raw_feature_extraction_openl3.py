#!/usr/bin/env python
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa  # for loading audio files
import soundfile as sf  # ensures mp3 decoding on some platforms
import torch
import torchopenl3

def extract_openl3_vector(  # helper that that takes a path to one mp3 and returns one vector for the whole track
    file_path: str,
    sr: int = 48000,    # openl3 models expect 48khz audio -> resample to that
    input_repr: str = "mel256", # spectogram frontend
    content_type: str = "music",    # mudic trained openl3 model should be used (better for songs)
    embedding_size: int = 512,  # output vector size per time frame
    hop_size: float = 0.5,  # analyze audio in 0.5 second window hops
    batch_size: int = 32,   # how many windows are embedded at once
    device: str = None, # gpu if available else cpu
):
    """
    Returns a (D,) mean-pooled OpenL3 embedding for a single track.
    """
    try:
        # Load mono and resample to 48k for OpenL3
        y, _ = librosa.load(file_path, sr=sr, mono=True)    # y: numpy array of samples        
        if y.size == 0:
            return None

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # key call: run openl3 on the waveform -> (T, D) embeddings over time windows
        emb, _ = torchopenl3.get_audio_embedding(
            y,
            sr,
            input_repr=input_repr,
            content_type=content_type,
            embedding_size=embedding_size,
            center=True,
            hop_size=hop_size,  # chops audio into short windows & moves forward by 0.5 seconds each step
            batch_size=batch_size,
        )   # returns emb = 2D array of embeddings, shape (T, D) : T -> number of time windows for this track, D -> embedding size (512); _ = timestamps (we don't need them)

        if emb is None or len(emb) == 0:
            return None

        # currently we have one vector per time slice, not per track yet
        # Maean-pool over time : average all T time slice vectors to get one fixed length vector per track with shape (D, ) -> single feature vector for each node (song)
        vec = emb.mean(axis=0)
        # convert pytorch tensor to numpy before casting dtype
        vec = vec.detach().cpu().numpy().astype(np.float32)
        return vec

    except Exception as e:
        print(f"[WARN] Failed on {file_path}: {e}")
        return None
    
# recursively iteration through audio directory, yields full paths to the mp3 files
def list_mp3s(audio_dir: str):
    for root, _, files in os.walk(audio_dir):
        for f in files:
            if f.lower().endswith(".mp3"):
                yield os.path.join(root, f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", default="../data/fma_small", help="Path to fma_small directory")
    ap.add_argument("--out_parquet", default="../data/fma_small_openl3.parquet",
                    help="Output Parquet file (features)")  # main output (table)
    ap.add_argument("--out_numpy", default="../data/fma_small_openl3.npy",
                    help="Optional NumPy .npy dump of the feature matrix")  # optional output: raw matrix
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--input_repr", choices=["mel256", "mel128", "linear"], default="mel256")
    ap.add_argument("--content_type", choices=["music", "env"], default="music")
    ap.add_argument("--embedding_size", type=int, choices=[512, 6144], default=512)
    ap.add_argument("--hop_size", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    paths = list(list_mp3s(args.audio_dir))
    print(f"Found {len(paths)} MP3 files under {args.audio_dir}")   # collect all mp3 paths and print count

    features = []   # to collect per track vectors
    idxs = []   # to colloect interger track_id matching tracks.csv
    skipped = 0 # count of files we couldn't process

    for p in tqdm(paths, desc="Extracting OpenL3"):
        # FMA filenames are zero-padded track IDs like 000123.mp3
        basename = os.path.splitext(os.path.basename(p))[0] # strip and cast to int -> 123 (row index when we create DataFrame, so it aligns with tracks.csv)
        try:
            track_id = int(basename)  # keep as int to match tracks.csv index
        except ValueError:
            skipped += 1
            continue

        vec = extract_openl3_vector(    # calls function above to get per track vector
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

        idxs.append(track_id)   # save vector and its track id (same order for both lists)
        features.append(vec)

    if not features:
        raise RuntimeError("No features extracted — check paths/codecs/dependencies.")

    X = np.vstack(features)  # stack all (D,) vectors into matrix X of shape (N, D) with N = number of tracks processed
    df = pd.DataFrame(X, index=pd.Index(idxs, name="track_id")) # build DataFrame with index = track_id (makes it easier to join with tracks.csv)
    df.to_parquet(args.out_parquet, index=True) # save features: compact, columnar, keeps index and good for later merges
    np.save(args.out_numpy, X)  # save features: raw matrix

    print(f"Saved {df.shape} features to {args.out_parquet}")
    print(f"Also wrote matrix to {args.out_numpy}")
    print(f"Skipped files: {skipped}")

if __name__ == "__main__":
    main()