"""
Simple inference script for hornet detection.

Usage:
    py -3.10 inference.py <path_to_audio_file>

Example:
    py -3.10 inference.py C:\path\to\audio.wav
"""

import sys
import numpy as np
import openl3
import librosa
import pickle
from pathlib import Path


def load_model_and_scaler():
    """Load trained model and scaler."""
    model_dir = Path(__file__).parent / "models"

    model_file = model_dir / "hornet_detector_final.pkl"
    scaler_file = model_dir / "scaler_final.pkl"

    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler


def extract_features(audio_path):
    """
    Extract OpenL3 features from audio file.

    Args:
        audio_path: Path to audio file (.wav, .flac, .mp3, etc.)

    Returns:
        embeddings: OpenL3 512-D embeddings for each 1-second window
    """
    print(f"Loading audio: {audio_path}")

    # Load audio at 48kHz (OpenL3 requirement)
    audio, sr = librosa.load(audio_path, sr=48000)
    duration = len(audio) / sr

    print(f"Audio duration: {duration:.2f} seconds")

    # Load OpenL3 model
    print("Loading OpenL3 model...")
    openl3_model = openl3.models.load_audio_embedding_model(
        input_repr="mel256",
        content_type="env",
        embedding_size=512
    )

    # Extract embeddings (1-second windows with 1-second hop)
    print("Extracting features...")
    embeddings, timestamps = openl3.get_audio_embedding(
        audio,
        sr,
        model=openl3_model,
        hop_size=1.0
    )

    print(f"Extracted {len(embeddings)} embeddings")

    return embeddings, timestamps


def predict_hornet(embeddings, model, scaler):
    """
    Predict if audio contains hornets.

    Args:
        embeddings: OpenL3 features
        model: Trained classifier
        scaler: Feature scaler

    Returns:
        predictions: 1 for hornet, -1 for non-hornet (per window)
        scores: Confidence scores (higher = more confident it's a hornet)
    """
    # Standardize features
    embeddings_scaled = scaler.transform(embeddings)

    # Predict
    predictions = model.predict(embeddings_scaled)
    scores = model.score_samples(embeddings_scaled)

    return predictions, scores


def main():
    """Main inference pipeline."""

    if len(sys.argv) < 2:
        print("Usage: py -3.10 inference.py <path_to_audio_file>")
        print("\nExample:")
        print("  py -3.10 inference.py C:\\Users\\Zdravkovic\\Downloads\\test_audio.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    if not Path(audio_path).exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    print("="*70)
    print("HORNET DETECTION")
    print("="*70)

    # Load model
    print("\nLoading trained model...")
    model, scaler = load_model_and_scaler()
    print("Model loaded!")

    # Extract features
    embeddings, timestamps = extract_features(audio_path)

    # Predict
    print("\nRunning detection...")
    predictions, scores = predict_hornet(embeddings, model, scaler)

    # Analyze results
    n_hornet = np.sum(predictions == 1)
    n_no_hornet = np.sum(predictions == -1)
    hornet_ratio = n_hornet / len(predictions)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nTotal audio windows analyzed: {len(predictions)}")
    print(f"Windows WITH hornet sounds: {n_hornet} ({hornet_ratio*100:.1f}%)")
    print(f"Windows WITHOUT hornet sounds: {n_no_hornet} ({(1-hornet_ratio)*100:.1f}%)")
    print(f"\nAverage confidence score: {np.mean(scores):.3f}")
    print(f"(Higher score = more confident it's a hornet)")

    # Overall verdict
    print("\n" + "="*70)
    if hornet_ratio >= 0.5:
        print("VERDICT: HORNET DETECTED ✓")
        print(f"Confidence: {hornet_ratio*100:.1f}% of audio matches hornet patterns")
    else:
        print("VERDICT: NO HORNET DETECTED")
        print(f"Only {hornet_ratio*100:.1f}% of audio matches hornet patterns")
    print("="*70)

    # Detailed timeline (optional, shows when hornets detected)
    if len(predictions) <= 20:  # Only show for short clips
        print("\nDetailed timeline:")
        for i, (t, pred, score) in enumerate(zip(timestamps, predictions, scores)):
            status = "HORNET" if pred == 1 else "no hornet"
            print(f"  {t:.1f}s: {status} (score: {score:.3f})")


if __name__ == "__main__":
    main()
