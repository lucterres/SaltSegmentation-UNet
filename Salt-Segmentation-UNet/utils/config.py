import os

import torch

# ---------------------------------------------------------------------------
# Dataset paths — TGS Salt Identification Challenge
# ---------------------------------------------------------------------------
TGS_PATH           = r'D:\dataset\tgs-salt\train'
IMAGE_DATASET_PATH = os.path.join(TGS_PATH, 'images')
MASK_DATASET_PATH  = os.path.join(TGS_PATH, 'masks')

SYNTH_IMAGE_PATH = os.path.join('dataset', 'synthetic', 'images')
SYNTH_MASK_PATH  = os.path.join('dataset', 'synthetic', 'masks')

# Fixed test-set path list (written once, shared across all runs)
TEST_PATHS = os.path.join('output', 'test_paths.txt')

# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
TEST_SPLIT = 0.20   # 800 test / 3200 train (from ~4000 total)
VAL_SPLIT  = 0.10   # 10% of train → validation during training

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
PIN_MEMORY = DEVICE == 'cuda'

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
NUM_CHANNELS = 1   # TGS is grayscale
NUM_CLASSES  = 1

# U-Net channel progression: (in, enc1, enc2, enc3)
ENCODER_CHANNELS = (1, 16, 32, 64)
DECODER_CHANNELS = (64, 32, 16)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
LR         = 1e-4
EPOCHS     = 50
BATCH_SIZE = 16
THRESHOLD  = 0.5
EARLY_STOP_PATIENCE = 10  # stop if val-IoU doesn't improve for N epochs

# ---------------------------------------------------------------------------
# Input size  (101 → pad/resize to 128 for clean MaxPool2d divisions)
# ---------------------------------------------------------------------------
INPUT_IMAGE_HEIGHT = 128
INPUT_IMAGE_WIDTH  = 128

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
BASE_OUTPUT = 'output'
PLOT_PATH   = os.path.join(BASE_OUTPUT, 'plot.png')
