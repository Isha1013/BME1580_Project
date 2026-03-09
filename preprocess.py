import pandas as pd
import numpy as np
import librosa
import os
from pathlib import Path
from tqdm import tqdm


AUDIO_DIR = Path("archive/Audio Files") # folder with .wav files
METADATA_FILE = "audio_metadata.csv"   # metadata CSV
OUTPUT_FILE = "audio_features.csv"     # CSV with extracted features

TARGET_SR = 4000   # resample rate
N_MFCC = 13        # number of MFCC coefficients


def extract_features(y, sr):
    """
    Compute standard audio features for traditional ML
    """

    # MFCCs (mean and std for each coeff)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_mean = np.mean(bandwidth)
    bandwidth_std = np.std(bandwidth)

    # Root Mean Square Energy
    rmse = librosa.feature.rms(y=y)
    rmse_mean = np.mean(rmse)
    rmse_std = np.std(rmse)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=4)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # Spectral Flatness
    flatness = librosa.feature.spectral_flatness(y=y)
    flatness_mean = np.mean(flatness)
    flatness_std = np.std(flatness)

    # Onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_mean = np.mean(onset_env)
    onset_std = np.std(onset_env)

    # Combine all features into dict
    features = {}

    for i in range(N_MFCC):
        features[f"mfcc{i+1}_mean"] = mfccs_mean[i]
        features[f"mfcc{i+1}_std"] = mfccs_std[i]

    for i in range(len(contrast_mean)):
        features[f"contrast{i+1}_mean"] = contrast_mean[i]
        features[f"contrast{i+1}_std"] = contrast_std[i]

    for i in range(len(chroma_mean)):
        features[f"chroma{i+1}_mean"] = chroma_mean[i]
        features[f"chroma{i+1}_std"] = chroma_std[i]

    features.update({
        "zcr_mean": zcr_mean,
        "zcr_std": zcr_std,
        "centroid_mean": centroid_mean,
        "centroid_std": centroid_std,
        "rolloff_mean": rolloff_mean,
        "rolloff_std": rolloff_std,
        "bandwidth_mean": bandwidth_mean,
        "bandwidth_std": bandwidth_std,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "flatness_mean": flatness_mean,
        "flatness_std": flatness_std,
        "onset_mean": onset_mean,
        "onset_std": onset_std  
    })

    return features


# Audio preprocessing
def preprocess_audio(file_path, target_sr=TARGET_SR):
    y, sr = librosa.load(file_path, sr=None)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # normalize amplitude
    y = y / np.max(np.abs(y)) 
    return y, target_sr

def main():
    df = pd.read_csv(METADATA_FILE)
    feature_rows = []

    for _, row in df.iterrows():
    # for _, row in tqdm(df.iterrows(), total=len(df)):
        file_name = row["filename"]
        file_path = AUDIO_DIR / file_name

        if not file_path.exists():
            continue

        # preprocess
        y, sr = preprocess_audio(file_path)

        # extract features
        features = extract_features(y, sr)

        # include metadata and class labels
        features["filename"] = file_name
        features["sound_class"] = row["sound_class"]
        features["disease_class"] = row["disease_class"]
        features["patient_id"] = row.get("patient_id", None)

        feature_rows.append(features)

    # save features
    df_features = pd.DataFrame(feature_rows)
    df_features.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved features to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
