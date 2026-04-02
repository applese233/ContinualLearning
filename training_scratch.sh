export WANDB_API_KEY=
wandb login

conda create -n ConLearning python=3.10 -y
conda activate ConLearning
cd ContinueLearning
pip install -e .
pip install flash-attn==2.8.2 --no-build-isolation
# or downloading wheels from https://github.com/Dao-AILab/flash-attention/releases
# pip install flash_attn-2.8.2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
python3 scripts/prepare_two_stage_datasets.py --output_dir data/multi_stage_grpo
# PREPARE_DATA=0 bash examples/qwen2_5_7b_multi_stage_grpo.sh
# STAGE_SEQUENCE=code,math bash examples/qwen2_5_7b_multi_stage_grpo.sh
# STAGE_SEQUENCE=math,code,math bash examples/qwen2_5_7b_multi_stage_grpo.sh
# STAGE1_TOTAL_EPOCHS=1 STAGE2_TOTAL_EPOCHS=3 STAGE2_ACTOR_LR=3.0e-7 bash examples/qwen2_5_7b_multi_stage_grpo.sh