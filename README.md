## Vision-SR1: Self-Rewarding Vision-Language Model via Reasoning Decomposition

[[📖 Paper](---)]  

**Models:**  
[🤗 Vision-SR1-7B](https://huggingface.co/LMMs-Lab-Turtle/SelfRewarded-R1-7B) | 
[🤗 Vision-SR1-7B-Cold-Start](https://huggingface.co/LMMs-Lab-Turtle/Qwen-2.5VL-7B-Cold-Start) |
[🤗 Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) 

**Datasets:**  
[📊 Vision-SR1-Cold-Start-9K](https://huggingface.co/datasets/LMMs-Lab-Turtle/Vision-SR1-Cold-9K)  | 
[📊 Vision-SR1-47K](https://huggingface.co/datasets/LMMs-Lab-Turtle/Vision-SR1-47K) 

**Training Curves:**  
[📈 Vision-SR1](https://api.wandb.ai/links/zli12321-university-of-maryland/85ed11ft) 

---

## Quick Start

### Environment Setup

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate vision-sr1-rl

# 2. Install dependencies
bash setup.sh
```

See [SETUP.md](SETUP.md) for detailed setup instructions.

### Training

```bash
# Qwen3-VL-8B Self-Reward GRPO Training
bash ./train_examples/start_test_train.sh

# Or use interactive script
bash ./train_examples/test_qwen3vl_8b_train.sh
```

**Note**: The training configuration uses Qwen3-VL-8B model. See `train_examples/test_qwen3vl_8b_config.yaml` for details.

### Data Preparation

Convert data to JSONL format:

```bash
python scripts/convert_test_data.py \
    --input_dir /path/to/data \
    --output_train ./data/train.jsonl \
    --output_val ./data/val.jsonl \
    --train_ratio 0.9
```

## Requirements

- Python 3.11
- CUDA 12.4+ (11.8, 12.1 also supported)
- 4-8 GPUs (40GB+ per GPU for 7B/8B model)
- **For Qwen3-VL-8B**: transformers==4.49.0, vllm==0.8.4

See [SETUP.md](SETUP.md) for detailed version requirements.

## Citation

```bibtex
@misc{li2025selfrewardingvisionlanguagemodelreasoning,
      title={Self-Rewarding Vision-Language Model via Reasoning Decomposition}, 
      author={Zongxia Li and Wenhao Yu and Chengsong Huang and Rui Liu and Zhenwen Liang and Fuxiao Liu and Jingxi Che and Dian Yu and Jordan Boyd-Graber and Haitao Mi and Dong Yu},
      year={2025},
      eprint={2508.19652},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.19652}, 
}
```

---

**Author**: zywang
