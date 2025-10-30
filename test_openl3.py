
import numpy as np
import openl3

print("=" * 60)
print("Testing OpenL3 Installation")
print("=" * 60)


try:
    model = openl3.models.load_audio_embedding_model(
        input_repr="mel256",
        content_type="env",
        embedding_size=512
    )
    print("\n[SUCCESS] Model loaded successfully!")
except Exception as e:
    print(f"\n[ERROR] Error loading model: {e}")
    exit(1)

# Test with dummy audio (1 second at 48kHz)
print("\nGenerating dummy audio (1 second at 48kHz)...")
sample_rate = 48000
dummy_audio = np.random.randn(sample_rate).astype(np.float32)

# Extract embeddings
print("Extracting embeddings...")
try:
    embeddings, timestamps = openl3.get_audio_embedding(
        dummy_audio,
        sample_rate,
        model=model,
        hop_size=0.5  # Extract embeddings every 0.5 seconds
    )
    print(f"\n[SUCCESS] Embeddings extracted successfully!")
    print(f"  - Embedding shape: {embeddings.shape}")
    print(f"  - Number of time steps: {len(timestamps)}")
    print(f"  - Timestamps: {timestamps}")
    print(f"  - Embedding dimensions: {embeddings.shape[1]}")

except Exception as e:
    print(f"\n[ERROR] Error extracting embeddings: {e}")
    exit(1)


