"""
Extract OpenL3 audio embeddings from preprocessed clips.
"""

import numpy as np
import openl3
from pathlib import Path
from tqdm import tqdm
import json

def extract_openl3_embeddings(processed_data_dir, output_dir, model=None):
    """
    Extract OpenL3 embeddings from all preprocessed clips.

    Args:
        processed_data_dir: Directory containing preprocessed .npz files
        output_dir: Where to save extracted embeddings
        model: Pre-loaded OpenL3 model (optional, will load if None)
    """
    processed_data_dir = Path(processed_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load OpenL3 model if not provided
    if model is None:
        print("Loading OpenL3 model (mel256, env, 512-D)...")
        model = openl3.models.load_audio_embedding_model(
            input_repr="mel256",
            content_type="env",
            embedding_size=512
        )
        print("Model loaded successfully!")

    # Find all preprocessed clip files
    clip_files = sorted(list(processed_data_dir.glob("**/*.npz")))
    print(f"\nFound {len(clip_files)} clips to process")

    if len(clip_files) == 0:
        print("No clips found! Check the processed_data_dir path.")
        return

    embeddings_list = []
    metadata_list = []

    # Process each clip
    for clip_file in tqdm(clip_files, desc="Extracting embeddings"):
        # Load preprocessed clip
        data = np.load(clip_file)

        # We have mel-spectrograms, but OpenL3 expects raw audio
        # So we need to load the original audio and extract with OpenL3
        audio_path = str(data['audio_path'])
        timestamp = float(data['timestamp'])

        # Load 1-second audio clip at the timestamp
        import librosa
        audio, sr = librosa.load(
            audio_path,
            sr=48000,  # OpenL3 expects 48kHz
            offset=timestamp,
            duration=1.0
        )

        # Extract OpenL3 embedding
        # Note: OpenL3 will return embeddings for 0.5s windows
        # For a 1-second clip, we'll get 2 embeddings
        emb, ts = openl3.get_audio_embedding(
            audio,
            sr,
            model=model,
            hop_size=1.0  # Use 1.0 to get single embedding per clip
        )

        # Take the mean if we get multiple embeddings
        if emb.shape[0] > 1:
            emb = np.mean(emb, axis=0, keepdims=True)

        embeddings_list.append(emb[0])  # Shape: (512,)

        # Store metadata
        metadata_list.append({
            'clip_file': str(clip_file.relative_to(processed_data_dir)),
            'audio_path': audio_path,
            'video_path': str(data['video_path']),
            'timestamp': timestamp,
            'pair_id': clip_file.parent.name,  # e.g., "pair_0"
        })

    # Convert to arrays
    embeddings = np.array(embeddings_list)  # Shape: (n_clips, 512)

    print(f"\nExtracted embeddings shape: {embeddings.shape}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Number of clips: {embeddings.shape[0]}")

    # Save embeddings and metadata
    embeddings_file = output_dir / "embeddings.npz"
    metadata_file = output_dir / "metadata.json"

    np.savez_compressed(
        embeddings_file,
        embeddings=embeddings
    )

    with open(metadata_file, 'w') as f:
        json.dump(metadata_list, f, indent=2)

    print(f"\nEmbeddings saved to: {embeddings_file}")
    print(f"Metadata saved to: {metadata_file}")

    return embeddings, metadata_list


def main():
    """Extract features from all datasets."""

    # Load OpenL3 model once (to avoid reloading for each dataset)
    print("=" * 70)
    print("OpenL3 Feature Extraction")
    print("=" * 70)

    print("\nLoading OpenL3 model...")
    model = openl3.models.load_audio_embedding_model(
        input_repr="mel256",
        content_type="env",
        embedding_size=512
    )
    print("Model loaded!\n")

    # Process 1-min dataset
    print("=" * 70)
    print("Extracting features from 1-min dataset")
    print("=" * 70)

    embeddings_1min, metadata_1min = extract_openl3_embeddings(
        processed_data_dir=r"C:\Users\Zdravkovic\Desktop\hornet_detection\processed_data\1min",
        output_dir=r"C:\Users\Zdravkovic\Desktop\hornet_detection\features\1min",
        model=model
    )

    # Process 5-min dataset
    print("\n" + "=" * 70)
    print("Extracting features from 5-min dataset")
    print("=" * 70)

    embeddings_5min, metadata_5min = extract_openl3_embeddings(
        processed_data_dir=r"C:\Users\Zdravkovic\Desktop\hornet_detection\processed_data\5min",
        output_dir=r"C:\Users\Zdravkovic\Desktop\hornet_detection\features\5min",
        model=model
    )

    # Combine all features
    print("\n" + "=" * 70)
    print("Combining all features")
    print("=" * 70)

    all_embeddings = np.vstack([embeddings_1min, embeddings_5min])
    all_metadata = metadata_1min + metadata_5min

    output_dir = Path(r"C:\Users\Zdravkovic\Desktop\hornet_detection\features\combined")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "embeddings.npz",
        embeddings=all_embeddings
    )

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\nCombined embeddings shape: {all_embeddings.shape}")
    print(f"Total clips: {len(all_metadata)}")
    print(f"Saved to: {output_dir}")

    print("\n" + "=" * 70)
    print("Feature extraction complete!")
    print("=" * 70)
    print(f"\nReady for training with {all_embeddings.shape[0]} samples")
    print(f"Each sample has {all_embeddings.shape[1]} features")


if __name__ == "__main__":
    main()
