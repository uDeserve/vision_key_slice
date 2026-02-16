# Setup Guide

## Environment Requirements

- Python 3.11
- CUDA 12.4+ (11.8, 12.1 also supported)
- Conda

## Installation Steps

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate vision-sr1-rl
```

### 2. Install Dependencies

```bash
bash setup.sh
```

The script will automatically:
- Detect CUDA version
- Install PyTorch 2.6.0 with matching CUDA
- Install flash-attention 2.7.4.post1
- Install vllm 0.8.4
- Install transformers 4.49.0
- Install project dependencies

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## Data Preparation

### Convert Data to JSONL

```bash
python scripts/convert_test_data.py \
    --input_dir /path/to/data \
    --output_train ./data/train.jsonl \
    --output_val ./data/val.jsonl \
    --train_ratio 0.9
```

### Data Format

Each line in JSONL should be:

```json
{
  "question": "Your prompt text",
  "ground_truth": "Expected response",
  "images": ["path/to/image1.jpg", "path/to/image2.jpg"]
}
```

## Training

### Start Training

```bash
# Non-interactive (recommended)
bash train_examples/start_test_train.sh

# Interactive
bash train_examples/test_qwen3vl_8b_train.sh
```

### Configuration

Edit `train_examples/test_qwen3vl_8b_config.yaml` to adjust:
- GPU count (`trainer.n_gpus_per_node`)
- Batch sizes (`data.rollout_batch_size`, `worker.actor.global_batch_size`)
- Model path (`worker.actor.model.model_path`)
- Data paths (`data.train_files`, `data.val_files`)

### Monitor Training

```bash
# View logs
tail -f logs/train_*.log

# Check GPU usage
watch -n 1 nvidia-smi
```

## Troubleshooting

### CUDA Version Mismatch

If CUDA version differs, modify `setup.sh` to use correct PyTorch index URL:
- CUDA 12.4+: `--index-url https://download.pytorch.org/whl/cu124`
- CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`
- CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`

### Flash Attention Installation

If pre-built wheel unavailable, install from source:
```bash
pip install flash-attn --no-build-isolation
```

### Ray Temporary Directory

If `/tmp` is full, set custom Ray temp directory:
```bash
export RAY_TMPDIR=/path/to/large/disk/ray_tmp
mkdir -p $RAY_TMPDIR
```

### HuggingFace Download Issues

The script automatically uses `https://hf-mirror.com` mirror. If issues persist, check network connectivity.

---

**Author**: zywang


