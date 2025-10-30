"""
Data preprocessing for hornet detection.
Handles audio-video synchronization and clip extraction.
"""

import os
import numpy as np
import soundfile as sf
import librosa
import cv2
from pathlib import Path
from tqdm import tqdm
import json


class HornetDataPreprocessor:
    """Preprocesses audio-visual hornet data."""

    def __init__(self, data_root, output_dir, sample_rate=48000, clip_duration=1.0, overlap=0.5):
        """
        Args:
            data_root: Path to folder containing video and audio pairs
            output_dir: Where to save preprocessed data
            sample_rate: Audio sample rate (Hz)
            clip_duration: Length of each clip (seconds)
            overlap: Overlap between consecutive clips (seconds)
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.overlap = overlap
        self.stride = clip_duration - overlap

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_audio_video_pairs(self):
        """Find matching audio and video files."""
        audio_files = sorted(list(self.data_root.glob("**/*.flac")))
        video_files = sorted(list(self.data_root.glob("**/*.mp4")))

        pairs = []
        for audio in audio_files:
            # Match based on file naming convention
            # Pattern: prefix_audio.flac and prefix_video.mp4
            audio_stem = audio.stem.replace("_audio", "")

            for video in video_files:
                video_stem = video.stem.replace("_video", "")
                if audio_stem == video_stem:
                    pairs.append((audio, video))
                    break

        return pairs

    def extract_audio_features(self, audio_path, start_time, duration):
        """
        Extract mel-spectrogram features from audio clip.

        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds
            duration: Duration of clip in seconds

        Returns:
            mel_spec: Mel-spectrogram (n_mels x time_steps)
        """
        # Load audio segment
        y, sr = librosa.load(
            audio_path,
            sr=self.sample_rate,
            offset=start_time,
            duration=duration
        )

        # Compute mel-spectrogram (similar to OpenL3's input)
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=256,  # Same as OpenL3's mel256
            fmin=0,
            fmax=sr // 2
        )

        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        return log_mel_spec

    def extract_video_frame(self, video_path, time_sec):
        """
        Extract a single frame from video at specified time.

        Args:
            video_path: Path to video file
            time_sec: Time in seconds

        Returns:
            frame: RGB image (224 x 224 x 3)
        """
        cap = cv2.VideoCapture(str(video_path))

        # Get video FPS
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate frame number
        frame_num = int(time_sec * fps)

        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to 224x224
        frame = cv2.resize(frame, (224, 224))

        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0

        return frame

    def process_pair(self, audio_path, video_path, sync_offset=0.0):
        """
        Process a single audio-video pair into clips.

        Args:
            audio_path: Path to audio file
            video_path: Path to video file
            sync_offset: Time offset between audio and video (seconds)
                        Positive means audio is ahead of video

        Returns:
            clips: List of (audio_features, video_frame, timestamp) tuples
        """
        # Get audio duration
        audio_info = sf.info(str(audio_path))
        duration = audio_info.duration

        clips = []

        # Extract clips with overlap
        current_time = 0
        while current_time + self.clip_duration <= duration:
            # Extract audio features
            audio_features = self.extract_audio_features(
                audio_path,
                current_time,
                self.clip_duration
            )

            # Extract corresponding video frame (middle of audio clip)
            # Apply sync offset
            video_time = current_time + (self.clip_duration / 2) - sync_offset

            if video_time >= 0:  # Make sure we don't go negative
                frame = self.extract_video_frame(video_path, video_time)

                if frame is not None:
                    clips.append({
                        'audio': audio_features,
                        'video': frame,
                        'timestamp': current_time,
                        'audio_path': str(audio_path),
                        'video_path': str(video_path)
                    })

            current_time += self.stride

        return clips

    def save_clips(self, clips, pair_id):
        """Save processed clips to disk."""
        pair_dir = self.output_dir / f"pair_{pair_id}"
        pair_dir.mkdir(exist_ok=True)

        for i, clip in enumerate(clips):
            clip_path = pair_dir / f"clip_{i:04d}.npz"
            np.savez_compressed(
                clip_path,
                audio=clip['audio'],
                video=clip['video'],
                timestamp=clip['timestamp'],
                audio_path=clip['audio_path'],
                video_path=clip['video_path']
            )

    def process_all(self, sync_offsets=None):
        """
        Process all audio-video pairs.

        Args:
            sync_offsets: Dict mapping pair_id to sync offset
                         If None, assumes perfect sync
        """
        pairs = self.find_audio_video_pairs()

        if not pairs:
            print("No audio-video pairs found!")
            return

        print(f"Found {len(pairs)} audio-video pairs")

        if sync_offsets is None:
            sync_offsets = {i: 0.0 for i in range(len(pairs))}

        all_metadata = []

        for pair_id, (audio_path, video_path) in enumerate(tqdm(pairs, desc="Processing pairs")):
            print(f"\nProcessing pair {pair_id}: {audio_path.name} + {video_path.name}")

            offset = sync_offsets.get(pair_id, 0.0)
            clips = self.process_pair(audio_path, video_path, sync_offset=offset)

            print(f"  Extracted {len(clips)} clips")

            self.save_clips(clips, pair_id)

            all_metadata.append({
                'pair_id': pair_id,
                'audio_file': str(audio_path),
                'video_file': str(video_path),
                'sync_offset': offset,
                'num_clips': len(clips)
            })

        # Save metadata
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(all_metadata, f, indent=2)

        total_clips = sum(m['num_clips'] for m in all_metadata)
        print(f"\nTotal clips extracted: {total_clips}")
        print(f"Data saved to: {self.output_dir}")


def main():
    """Example usage."""
    # Process 1-min dataset
    print("=" * 50)
    print("Processing 1-min dataset")
    print("=" * 50)

    preprocessor_1min = HornetDataPreprocessor(
        data_root=r"C:\Users\Zdravkovic\Downloads\1-min-no-time-clapper-30fps\1-min-no-time-clapper-30fps",
        output_dir=r"C:\Users\Zdravkovic\Desktop\hornet_detection\processed_data\1min",
        clip_duration=1.0,
        overlap=0.5
    )

    # Sync offsets determined by inspection - all <0.15 seconds, using 0.0
    sync_offsets_1min = {
        0: 0.0,  # pair 0 - measured 0.133s difference (negligible)
        1: 0.0,  # pair 1 - measured 0.100s difference (negligible)
        2: 0.0,  # pair 2 - measured 0.100s difference (negligible)
    }

    preprocessor_1min.process_all(sync_offsets=sync_offsets_1min)

    # Process 5-min dataset
    print("\n" + "=" * 50)
    print("Processing 5-min dataset")
    print("=" * 50)

    preprocessor_5min = HornetDataPreprocessor(
        data_root=r"C:\Users\Zdravkovic\Downloads\5-min-time-clapper-45fps\5-min-time-clapper-45fps",
        output_dir=r"C:\Users\Zdravkovic\Desktop\hornet_detection\processed_data\5min",
        clip_duration=1.0,
        overlap=0.5
    )

    # Sync offsets determined by inspection - excellent sync
    sync_offsets_5min = {
        0: 0.0,  # pair 0 - measured 0.044s difference (excellent)
        1: 0.0,  # pair 1 - measured 0.044s difference (excellent)
    }

    preprocessor_5min.process_all(sync_offsets=sync_offsets_5min)


if __name__ == "__main__":
    main()
