import sys
import torch
import hashlib
from pathlib import Path

sys.path.insert(0, "F:/deeplearning/old_photo_restoration_blueprint21_submission/src")
from old_photo_restoration.segmentation.model_r014 import CrackSegmenterR014ResNet34

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    checkpoint_path = "F:/deeplearning/experiment_value/r014_resnet34_safe_experiment/kaggle_outputs/r014_resnet34_outputs_download/checkpoints/stage2_real_best_val_iou.pth"
    expected_sha256 = "e1d84ced2e3aac6fd89bbe48bd6149cc445cc7308b03887ac3f66de2352924c2"
    
    actual_sha256 = get_sha256(checkpoint_path)
    if actual_sha256 != expected_sha256:
        print(f"SHA256 mismatch! Expected {expected_sha256}, got {actual_sha256}")
        sys.exit(1)
    
    print("SHA256 matched.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrackSegmenterR014ResNet34(pretrained=False).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Check for missing/unexpected keys
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    
    model.eval()
    
    dummy_x = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        out = model(dummy_x)
        
    if out.shape != (1, 1, 512, 512):
        print(f"Shape mismatch! Expected (1, 1, 512, 512), got {out.shape}")
        sys.exit(1)
        
    print("Shape test passed successfully. Output:", out.shape)
    
if __name__ == "__main__":
    main()
