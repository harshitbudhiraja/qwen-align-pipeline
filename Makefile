run-pipeline:
	python run_pipeline.py \
    --dataset data/synthetic_data.jsonl \
    --model Qwen/Qwen2.5-Coder-0.5B \
    --rl-method sft_only


run-grpo:
	python src/train_grpo.py --sft-checkpoint outputs/sft/final


run-sft: 
	python src/train_sft.py --model Qwen/Qwen2.5-Coder-0.5B --data data/sft_train.jsonl

