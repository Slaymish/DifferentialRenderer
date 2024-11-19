import torch

# Path to save rendered images
MODEL_SAVE_DIR = "model_checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


