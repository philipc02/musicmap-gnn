import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# mel-frequency cepstral coefficients -> captures short term power spectrum of a sound by representing on a Mel-scale of frequency
# Mel-scale: perceptual scale of frequencies based on human hearing -> equal distances in Mel-values = equal pitch differences
# cepstrum: representation of spectrum of a signal (take inverse FT of log of power spectrum)
# MFCC: compute Mel-spectogram (spectrum of signal on Mel-scale), apply discrete cosine transform
def extract_mfcc(file_path, sr=22050, n_mfcc=20):   # extracts 20 mfccs
    """
    Extract MFCC features from an audio file and return as mean-pooled vector.
    """
    try:
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)  # Mean pooling: https://media.geeksforgeeks.org/wp-content/uploads/20190721030705/Screenshot-2019-07-21-at-3.05.56-AM.png
        return mfcc_mean    # take mean over time of the 20 extracted mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    
def process_fma_small(audio_dir, output_csv="../data/fma_small_mfcc.csv"):
    """
    Process all MP3 files in FMA small dataset and save MFCC embeddings.
    """
    features = []
    track_ids = []

    for root, _, files in os.walk(audio_dir):
        for file in tqdm(files, desc="Processing audio"):   # show progress bar for this using tqdm
            if file.endswith(".mp3"):
                track_id = os.path.splitext(file)[0]
                file_path = os.path.join(root, file)

                mfcc_vector = extract_mfcc(file_path)   # run function above to extract mfccs
                if mfcc_vector is not None:
                    features.append(mfcc_vector)
                    track_ids.append(track_id)

    features_array = np.array(features) # we have created our own vector of audio features for each track from the raw audio!
    df = pd.DataFrame(features_array, index=track_ids)
    df.to_csv(output_csv)   # convert the feature array into a csv file that can be easily processed by our model
    # each row = track ID, each column: mfcc dimension

    print(f"Saved features to {output_csv} with shape {features_array.shape}")

if __name__ == "__main__":
    process_fma_small("../data/fma_small")