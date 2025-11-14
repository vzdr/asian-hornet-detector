"""Check what's in the saved checkpoint"""
import torch
from pathlib import Path

model_path = Path(__file__).parent / "models" / "best_ultimate_efficientnet.pth"

checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

print("Checkpoint contents:")
print("="*70)
for key in checkpoint.keys():
    if key != 'model_state_dict' and key != 'optimizer_state_dict':
        print(f"{key}: {checkpoint[key]}")
