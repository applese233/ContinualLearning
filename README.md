# EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)](https://github.com/hiyouga/EasyR1/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/llamafactory_ai)](https://twitter.com/llamafactory_ai)
[![Docker Pulls](https://img.shields.io/docker/pulls/hiyouga/verl)](https://hub.docker.com/r/hiyouga/verl/tags)

### Used by [Amazon Web Services](https://aws.amazon.com/cn/blogs/china/building-llm-model-hub-based-on-llamafactory-and-easyr1/)

This project is a clean fork of the original [veRL](https://github.com/volcengine/verl) project to support vision language models, we thank all the authors for providing such a high-performance RL training framework.

EasyR1 is efficient and scalable due to the design of **[HybirdEngine](https://arxiv.org/abs/2409.19256)** and the latest release of **[vLLM](https://github.com/vllm-project/vllm)**'s SPMD mode.

## Features

- Supported models
  - Llama3/Qwen2/Qwen2.5/Qwen3 language models
  - Qwen2-VL/Qwen2.5-VL/Qwen3-VL vision language models
  - DeepSeek-R1 distill models

- Supported algorithms
  - GRPO
  - DAPO ![new](https://img.shields.io/badge/new-orange)
  - Reinforce++
  - ReMax
  - RLOO
  - GSPO ![new](https://img.shields.io/badge/new-orange)
  - CISPO ![new](https://img.shields.io/badge/new-orange)

- Supported datasets
  - Any text, vision-text dataset in a [specific format](#custom-dataset)

- Supported tricks
  - Padding-free training
  - LoRA training ![new](https://img.shields.io/badge/new-orange)
  - Resuming from the latest/best checkpoint
  - Wandb & SwanLab & Mlflow & Tensorboard tracking

## Requirements

### Software Requirements

- Python 3.9+
- transformers>=4.54.0
- flash-attn>=2.4.3
- vllm>=0.8.3

We provide a [Dockerfile](./Dockerfile) to easily build environments.

We recommend using the [pre-built docker image](https://hub.docker.com/r/hiyouga/verl) in EasyR1.

```bash
docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
```

If your environment does not support Docker, you can consider using **Apptainer**:

```bash
apptainer pull easyr1.sif docker://hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
apptainer shell --nv --cleanenv --bind /mnt/your_dir:/mnt/your_dir easyr1.sif
```

Use `USE_MODELSCOPE_HUB=1` to download models from the ModelScope hub.

### Hardware Requirements

\* *estimated*

| Method                   | Bits |  1.5B  |   3B   |   7B   |   32B   |   72B   |
| ------------------------ | ---- | ------ | ------ | ------ | ------- | ------- |
| GRPO Full Fine-Tuning    |  AMP | 2*24GB | 4*40GB | 8*40GB | 16*80GB | 32*80GB |
| GRPO Full Fine-Tuning    | BF16 | 1*24GB | 1*40GB | 4*40GB |  8*80GB | 16*80GB |
| GRPO LoRA Fine-Tuning    |  AMP | 1*12GB | 1*24GB | 2*32GB |  2*80GB |  4*80GB |

> [!NOTE]
> Use `worker.actor.fsdp.torch_dtype=bf16` and `worker.actor.optim.strategy=adamw_bf16` to enable bf16 training.

## Tutorial: Run Qwen2.5-VL GRPO on [Geometry3K](https://huggingface.co/datasets/hiyouga/geometry3k) Dataset in Just 3 Steps

![image](assets/qwen2_5_vl_7b_geo.png)

### Installation

```bash
git clone https://github.com/hiyouga/EasyR1.git
cd EasyR1
pip install -e .
```

The command above is sufficient only if your current Python environment already contains a compatible stack for PyTorch, vLLM, FlashAttention and Transformers. For RL training, especially for custom GRPO workflows, we strongly recommend using a clean environment.

### Installation Without Docker or Apptainer

You do not need `sudo` to run EasyR1. A user-local Conda environment is enough if your machine already has a working NVIDIA driver and CUDA runtime.

1. Create and activate a clean Conda environment.

```bash
conda create -n ConLearn python=3.10 -y
conda activate ConLearn
python -m pip install --upgrade pip setuptools wheel
```

2. Install EasyR1 directly.

`requirements.txt` is pinned to a coherent runtime stack, so a fresh environment can install the project directly:

```bash
pip install -e .
```

3. Verify the runtime before starting training.

```bash
python -c "import torch, transformers, vllm; print(torch.__version__, transformers.__version__, vllm.__version__)"
python -c "import flash_attn; print('flash-attn ok')"
```

4. Prepare your datasets and launch training.

```bash
python scripts/prepare_two_stage_datasets.py --output_dir data/multi_stage_grpo
bash examples/qwen2_5_7b_multi_stage_grpo.sh
```

If you only want to prepare a subset, pass `--datasets` with comma- or space-separated names:

```bash
python scripts/prepare_two_stage_datasets.py \
  --output_dir data/multi_stage_grpo \
  --datasets "aime24 aime25 amc math500 minerva olympiadbench humanevalplus taco"
```

The default data prep now builds:

- `deepscaler_train.parquet` and `deepscaler_val.parquet` from `agentica-org/DeepScaleR-Preview-Dataset` for the math stage.
- `deepcoder_train.parquet` and `deepcoder_val.parquet` from `agentica-org/DeepCoder-Preview-Dataset` for the coding stage.
- Additional math evaluation sets such as AIME24, AIME25, AMC, MATH-500, Minerva Math, and OlympiadBench.
- Additional coding evaluation sets such as HumanEval, HumanEval+, LiveCodeBench, MBPP, APPS, and TACO.

If you want to cap the new primary train/validation sets during preparation, you can pass limits explicitly:

```bash
python scripts/prepare_two_stage_datasets.py \
  --output_dir data/multi_stage_grpo \
  --math500_val_limit 128 \
  --minerva_val_limit 64 \
  --olympiadbench_val_limit 64 \
  --aime24_val_limit 30 \
  --deepscaler_val_limit 500 \
  --deepcoder_val_limit 500
```

To run the direct vLLM evaluator on the expanded suite, the wrapper now accepts these benchmark names directly through `DATASETS`, for example:

```bash
DATASETS="aime24 aime25 amc math500 minerva olympiadbench humaneval humanevalplus livecodebench taco" \
CUDA_VISIBLE_DEVICES=4 \
bash examples/qwen2_5_7b_direct_eval.sh
```

The direct evaluator now keeps per-sample detail files smaller by default:

- `*_eval_results.json` keeps only a small sampled subset of detailed rows, controlled by `SAMPLE_DETAILS` (default `5`).
- full `*_generations.jsonl` exports are no longer written unless you explicitly opt in.
- for code benchmarks, the saved detail rows keep only a short ground-truth preview plus byte count instead of full hidden test payloads.

If you need the raw model output for debugging, enable it explicitly:

```bash
SAVE_ALL_GENERATIONS=1 SAVE_FULL_RESPONSE=1 bash examples/qwen2_5_7b_direct_eval.sh
```

Alternative stage schedules use the same training configs and only change stage orchestration:

```bash
STAGE_SEQUENCE=code,math bash examples/qwen2_5_7b_multi_stage_grpo.sh
STAGE_SEQUENCE=math,code,math bash examples/qwen2_5_7b_multi_stage_grpo.sh
BASE_MODEL_PATH=/path/to/hf_model bash examples/qwen2_5_7b_code_grpo.sh
STAGE1_TOTAL_EPOCHS=1 STAGE2_TOTAL_EPOCHS=3 STAGE2_ACTOR_LR=3.0e-7 bash examples/qwen2_5_7b_multi_stage_grpo.sh
STAGE3_TOTAL_EPOCHS=1 STAGE3_SAVE_FREQ=5 STAGE3_PRIMARY_VAL_LIMIT=64 bash examples/qwen2_5_7b_multi_stage_grpo.sh
```

For multi-stage runs, the launcher also accepts stage-specific overrides using the `STAGE{n}_...` pattern. For example, `STAGE2_MAX_RESPONSE_LENGTH=1024` only affects the second stage, while `TOTAL_EPOCHS=2` still acts as a global fallback for stages that do not define a stage-specific override.

When `PREPARE_DATA=1` is set on the launcher, it will prepare DeepScaleR and DeepCoder automatically before training.

Notes:

- If `flash-attn` fails to compile, the usual blocker is the local CUDA toolchain, not missing `sudo` privileges. In that case, keep the same Conda environment and retry with `pip install flash-attn==2.8.2 --no-build-isolation`.
- If `vllm` import fails, first check the installed `transformers` version. Mismatched `vllm` and `transformers` versions are a common cause of startup failures.
- Avoid reusing a large existing base environment with unrelated packages. Dependency drift is the most common reason for EasyR1 import errors.

### GRPO Full Training

```bash
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
```

### GRPO LoRA Training

```bash
bash examples/qwen3_vl_4b_geo3k_grpo_lora.sh
```

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

> [!TIP]
> If you encounter issues with connecting to Hugging Face, consider using `export HF_ENDPOINT=https://hf-mirror.com`.
>
> If you want to use SwanLab logger, consider using `bash examples/qwen2_5_vl_7b_geo3k_swanlab.sh`.

## Custom Dataset

Please refer to the example datasets to prepare your own dataset.

- Text dataset: https://huggingface.co/datasets/hiyouga/math12k
- Image-text dataset: https://huggingface.co/datasets/hiyouga/geometry3k
- Multi-image-text dataset: https://huggingface.co/datasets/hiyouga/journeybench-multi-image-vqa
- Text-image mixed dataset: https://huggingface.co/datasets/hiyouga/rl-mixed-dataset

## How to Understand GRPO in EasyR1

![image](assets/easyr1_grpo.png)

- To learn about the GRPO algorithm, you can refer to [Hugging Face's blog](https://huggingface.co/docs/trl/v0.16.1/en/grpo_trainer).

## How to Run 70B+ Model in Multi-node Environment

1. Start the Ray head node.

```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0
```

2. Start the Ray worker node and connect to the head node.

```bash
ray start --address=<head_node_ip>:6379
```

3. Check the Ray resource pool.

```bash
ray status
```

4. Run training script on the Ray head node only.

```bash
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
```

See the **[veRL's official doc](https://verl.readthedocs.io/en/latest/start/multinode.html)** for more details about multi-node training and Ray debugger.

## Other Baselines

We also reproduced the following two baselines of the [R1-V](https://github.com/deep-agent/R1-V) project.
- [CLEVR-70k-Counting](examples/baselines/qwen2_5_vl_3b_clevr.sh): Train the Qwen2.5-VL-3B-Instruct model on counting problem.
- [GeoQA-8k](examples/baselines/qwen2_5_vl_3b_geoqa8k.sh): Train the Qwen2.5-VL-3B-Instruct model on GeoQA problem.

## Performance Baselines

See [baselines.md](assets/baselines.md).

## Awesome Work using EasyR1

- **MMR1**: Enhancing Multimodal Reasoning with Variance-Aware Sampling and Open Resources. [![[code]](https://img.shields.io/github/stars/LengSicong/MMR1)](https://github.com/LengSicong/MMR1) [![[arxiv]](https://img.shields.io/badge/arxiv-2509.21268-blue)](https://arxiv.org/abs/2509.21268)
- **Vision-R1**: Incentivizing Reasoning Capability in Multimodal Large Language Models. [![[code]](https://img.shields.io/github/stars/Osilly/Vision-R1)](https://github.com/Osilly/Vision-R1) [![[arxiv]](https://img.shields.io/badge/arxiv-2503.06749-blue)](https://arxiv.org/abs/2503.06749)
- **Seg-Zero**: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement. [![[code]](https://img.shields.io/github/stars/dvlab-research/Seg-Zero)](https://github.com/dvlab-research/Seg-Zero) [![[arxiv]](https://img.shields.io/badge/arxiv-2503.06520-blue)](https://arxiv.org/abs/2503.06520)
- **MetaSpatial**: Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse. [![[code]](https://img.shields.io/github/stars/PzySeere/MetaSpatial)](https://github.com/PzySeere/MetaSpatial) [![[arxiv]](https://img.shields.io/badge/arxiv-2503.18470-blue)](https://arxiv.org/abs/2503.18470)
- **Temporal-R1**: Envolving Temporal Reasoning Capability into LMMs via Temporal Consistent Reward. [![[code]](https://img.shields.io/github/stars/appletea233/Temporal-R1)](https://github.com/appletea233/Temporal-R1) [![[arxiv]](https://img.shields.io/badge/arxiv-2506.01908-blue)](https://arxiv.org/abs/2506.01908)
- **NoisyRollout**: Reinforcing Visual Reasoning with Data Augmentation. [![[code]](https://img.shields.io/github/stars/John-AI-Lab/NoisyRollout)](https://github.com/John-AI-Lab/NoisyRollout) [![[arxiv]](https://img.shields.io/badge/arxiv-2504.13055-blue)](https://arxiv.org/pdf/2504.13055)
- **GUI-R1**: A Generalist R1-Style Vision-Language Action Model For GUI Agents. [![[code]](https://img.shields.io/github/stars/ritzz-ai/GUI-R1)](https://github.com/ritzz-ai/GUI-R1) [![[arxiv]](https://img.shields.io/badge/arxiv-2504.10458-blue)](https://arxiv.org/abs/2504.10458)
- **FAST-GRPO**: Fast-Slow Thinking framework that dynamically adapts reasoning depth based on question characteristics. [![[code]](https://img.shields.io/github/stars/Mr-Loevan/FAST)](https://github.com/Mr-Loevan/FAST) [![[arxiv]](https://img.shields.io/badge/arxiv-2504.18458-blue)](https://arxiv.org/abs/2504.18458)
- **R1-Track**: Direct Application of MLLMs to Visual Object Tracking via Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/Wangbiao2/R1-Track)](https://github.com/Wangbiao2/R1-Track)
- **VisionReasoner**: Unified Visual Perception and Reasoning via Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/dvlab-research/VisionReasoner)](https://github.com/dvlab-research/VisionReasoner) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.12081-blue)](https://arxiv.org/abs/2505.12081)
- **MM-UPT**: Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO. [![[code]](https://img.shields.io/github/stars/waltonfuture/MM-UPT)](https://github.com/waltonfuture/MM-UPT) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.22453-blue)](https://arxiv.org/pdf/2505.22453)
- **RL-with-Cold-Start**: Advancing Multimodal Reasoning via Reinforcement Learning with Cold Start. [![[code]](https://img.shields.io/github/stars/waltonfuture/RL-with-Cold-Start)](https://github.com/waltonfuture/RL-with-Cold-Start) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.22334-blue)](https://arxiv.org/pdf/2505.22334)
- **ViGoRL**: Grounded Reinforcement Learning for Visual Reasoning. [![[code]](https://img.shields.io/github/stars/Gabesarch/grounded-rl)](https://github.com/Gabesarch/grounded-rl) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.22334-blue)](https://arxiv.org/abs/2505.23678)
- **Revisual-R1**: Advancing Multimodal Reasoning: From Optimized Cold Start to Staged Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/CSfufu/Revisual-R1)](https://github.com/CSfufu/Revisual-R1) [![[arxiv]](https://img.shields.io/badge/arxiv-2506.04207-blue)](https://arxiv.org/abs/2506.04207)
- **SophiaVL-R1**: Reinforcing MLLMs Reasoning with Thinking Reward. [![[code]](https://img.shields.io/github/stars/kxfan2002/SophiaVL-R1)](https://github.com/kxfan2002/SophiaVL-R1) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.17018-blue)](https://arxiv.org/abs/2505.17018)
- **Vision-Matters**: Simple Visual Perturbations Can Boost Multimodal Math Reasoning. [![[code]](https://img.shields.io/github/stars/YutingLi0606/Vision-Matters)](https://github.com/YutingLi0606/Vision-Matters) [![[arxiv]](https://img.shields.io/badge/arxiv-2506.09736-blue)](https://arxiv.org/abs/2506.09736)
- **VTool-R1**: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use. [![[code]](https://img.shields.io/github/stars/VTOOL-R1/vtool-r1)](https://github.com/VTOOL-R1/vtool-r1) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.19255-blue)](https://arxiv.org/abs/2505.19255)
- **Long-RL**: Scaling RL to Long Sequences. [![[code]](https://img.shields.io/github/stars/NVlabs/Long-RL)](https://github.com/NVlabs/Long-RL) [![[arxiv]](https://img.shields.io/badge/arxiv-2507.07966-blue)](https://arxiv.org/abs/2507.07966)
- **EditGRPO**: Reinforcement Learning with Post-Rollout Edits for Clinically Accurate Chest X-Ray Report Generation. [![[code]](https://img.shields.io/github/stars/taokz/EditGRPO)](https://github.com/taokz/EditGRPO)
- **ARES**: Multimodal Adaptive Reasoning via Difficulty-Aware Token-Level Entropy Shaping. [![[code]](https://img.shields.io/github/stars/shawn0728/ARES)](https://github.com/shawn0728/ARES) [![[arxiv]](https://img.shields.io/badge/arxiv-2510.08457-blue)](https://arxiv.org/abs/2510.08457)
- **VPPO**: Spotlight on Token Perception for Multimodal Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/huaixuheqing/VPPO-RL)](https://github.com/huaixuheqing/VPPO-RL) [![[arxiv]](https://img.shields.io/badge/arxiv-2510.09285-blue)](https://arxiv.org/abs/2510.09285)
- **IE-Critic-R1**: Advancing the Explanatory Measurement of Text-Driven Image Editing for Human Perception Alignment. [![[code]](https://img.shields.io/github/stars/Coobiw/IE-Critic-R1)](https://github.com/Coobiw/IE-Critic-R1) [![[arxiv]](https://img.shields.io/badge/arxiv-2511.18055-blue)](https://arxiv.org/abs/2511.18055)
- **OneThinker**: All-in-one Reasoning Model for Image and Video. [![[code]](https://img.shields.io/github/stars/tulerfeng/OneThinker)](https://github.com/tulerfeng/OneThinker) [![[arxiv]](https://img.shields.io/badge/arxiv-2512.03043-blue)](https://arxiv.org/abs/2512.03043)
- **MetaphorStar**: Image Metaphor Understanding and Reasoning with End-to-End Visual Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/MING-ZCH/MetaphorStar)](https://github.com/MING-ZCH/MetaphorStar) [![[arxiv]](https://img.shields.io/badge/arxiv-2602.10575-blue)](https://arxiv.org/abs/2602.10575)

## TODO

- Support ulysses parallelism for VLMs (middle priority).
- Support more VLM architectures.

> [!NOTE]
> We will not provide scripts for supervised fine-tuning and inference in this project. If you have such requirements, we recommend using [LlamaFactory](https://github.com/hiyouga/LlamaFactory).

### Known bugs

These features are temporarily disabled for now, we plan to fix them one-by-one in the future updates.

- Vision language models are not compatible with ulysses parallelism yet.

## Discussion Group

👋 Join our [WeChat group](https://github.com/hiyouga/llamafactory-community/blob/main/wechat/easyr1.jpg).

## FAQs

> ValueError: Image features and image tokens do not match: tokens: 8192, features 9800

Increase the `data.max_prompt_length` or reduce the `data.max_pixels`.

> RuntimeError: CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:62

Reduce the `worker.rollout.gpu_memory_utilization` and enable `worker.actor.offload.offload_params`.

> RuntimeError: 0 active drivers ([]). There should only be one.

Uninstall `deepspeed` from the current python environment.

## Citation

Core contributors: [Yaowei Zheng](https://github.com/hiyouga), [Junting Lu](https://github.com/AL-377), [Shenzhi Wang](https://github.com/Shenzhi-Wang), [Zhangchi Feng](https://github.com/BUAADreamer), [Dongdong Kuang](https://github.com/Kuangdd01), Yuwen Xiong and Richong Zhang

We also thank Guangming Sheng and Chi Zhang for helpful discussions.

```bibtex
@misc{zheng2025easyr1,
  title        = {EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework},
  author       = {Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, Yuwen Xiong, Richong Zhang},
  howpublished = {\url{https://github.com/hiyouga/EasyR1}},
  year         = {2025}
}
```

We recommend to also cite the original work.

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```
