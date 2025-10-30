"""
Inspect hornet dataset files to understand their properties.
"""

import soundfile as sf
import cv2
from pathlib import Path

print("=" * 70)
print("Hornet Dataset Inspection")
print("=" * 70)

# 1-min dataset
print("\n### 1-MIN DATASET (no time clapper, 30fps)")
print("-" * 70)
dataset_1min = Path(r"C:\Users\Zdravkovic\Downloads\1-min-no-time-clapper-30fps\1-min-no-time-clapper-30fps")

audio_files_1min = sorted(list(dataset_1min.glob("**/*.flac")))
video_files_1min = sorted(list(dataset_1min.glob("**/*.mp4")))

print(f"\nFound {len(audio_files_1min)} audio files and {len(video_files_1min)} video files")

for i, (audio_path, video_path) in enumerate(zip(audio_files_1min, video_files_1min)):
    print(f"\n--- Pair {i+1} ---")
    print(f"Audio: {audio_path.name}")

    # Audio info
    audio_info = sf.info(str(audio_path))
    print(f"  Sample rate: {audio_info.samplerate} Hz")
    print(f"  Duration: {audio_info.duration:.2f} seconds")
    print(f"  Channels: {audio_info.channels}")
    print(f"  Format: {audio_info.format}, {audio_info.subtype}")

    print(f"\nVideo: {video_path.name}")

    # Video info
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Frame count: {frame_count}")
    print(f"  Resolution: {width}x{height}")

    # Check sync
    audio_video_diff = audio_info.duration - duration
    print(f"  Audio-Video duration difference: {audio_video_diff:.3f} seconds")

# 5-min dataset
print("\n\n### 5-MIN DATASET (with time clapper, 45fps)")
print("-" * 70)
dataset_5min = Path(r"C:\Users\Zdravkovic\Downloads\5-min-time-clapper-45fps\5-min-time-clapper-45fps")

audio_files_5min = sorted(list(dataset_5min.glob("**/*.flac")))
video_files_5min = sorted(list(dataset_5min.glob("**/*.mp4")))

print(f"\nFound {len(audio_files_5min)} audio files and {len(video_files_5min)} video files")

for i, (audio_path, video_path) in enumerate(zip(audio_files_5min, video_files_5min)):
    print(f"\n--- Pair {i+1} ---")
    print(f"Audio: {audio_path.name}")

    # Audio info
    audio_info = sf.info(str(audio_path))
    print(f"  Sample rate: {audio_info.samplerate} Hz")
    print(f"  Duration: {audio_info.duration:.2f} seconds")
    print(f"  Channels: {audio_info.channels}")
    print(f"  Format: {audio_info.format}, {audio_info.subtype}")

    print(f"\nVideo: {video_path.name}")

    # Video info
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Frame count: {frame_count}")
    print(f"  Resolution: {width}x{height}")

    # Check sync
    audio_video_diff = audio_info.duration - duration
    print(f"  Audio-Video duration difference: {audio_video_diff:.3f} seconds")

# Summary
print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

total_audio_duration = sum([sf.info(str(f)).duration for f in audio_files_1min + audio_files_5min])
print(f"\nTotal audio duration: {total_audio_duration:.2f} seconds ({total_audio_duration/60:.2f} minutes)")
print(f"Total pairs: {len(audio_files_1min) + len(audio_files_5min)}")
print(f"  1-min dataset: {len(audio_files_1min)} pairs")
print(f"  5-min dataset: {len(audio_files_5min)} pairs")

# Estimate number of 1-second clips with 0.5s overlap
clips_per_second = 1 / 0.5  # 2 clips per second with 0.5s stride
estimated_clips = int((total_audio_duration - 1) * clips_per_second) + 1
print(f"\nEstimated 1-second clips (0.5s overlap): ~{estimated_clips}")
