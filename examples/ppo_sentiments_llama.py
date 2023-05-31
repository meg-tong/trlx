# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function
import os
from typing import List
import argparse

import torch
from datasets import load_dataset
from transformers import pipeline

from trlx import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

from src.common import attach_debugger, is_main_process


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def llama_config(args):

    tokenizer_path_maybe = os.path.join(args.model, "tokenizer.model")
    tokenizer_path = tokenizer_path_maybe if os.path.exists(tokenizer_path_maybe) else args.model
    return TRLConfig(
        train=TrainConfig(
            seq_length=args.seq_length,
            epochs=args.epochs,
            total_steps=args.total_steps,
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint_interval,
            eval_interval=args.eval_interval,
            pipeline=args.pipeline,
            trainer=args.trainer,
            save_best=args.save_best,
            seed=args.seed,
        ),
        model=ModelConfig(model_path=args.model, num_layers_unfrozen=args.num_layers_unfrozen),
        tokenizer=TokenizerConfig(tokenizer_path=tokenizer_path, truncation_side="right", padding_side="left"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)
        ),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=args.T_max, eta_min=args.eta_min)),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=args.num_rollouts,
            chunk_size=args.chunk_size,
            ppo_epochs=args.ppo_epochs,
            init_kl_coef=args.init_kl_coef,
            target=args.target,
            horizon=args.horizon,
            gamma=args.gamma,
            lam=args.lam,
            cliprange=args.cliprange,
            cliprange_value=args.cliprange_value,
            vf_coef=args.vf_coef,
            scale_reward=args.scale_reward,
            ref_mean=args.ref_mean,
            ref_std=args.ref_std,
            cliprange_reward=args.cliprange_reward,
            gen_kwargs=dict(
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=args.do_sample,
            ),
        ),
    )


def main(args):
    # Merge sweep config with default config if given
    config = TRLConfig.from_dict(llama_config(args).to_dict())

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=device,
    )

    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return sentiments

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("imdb", split="train+test")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]

    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["I don't know much about Hungarian underground"] * 64,
        config=config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data/public_models/llama/llama_hf_weights/llama-7b")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", type=int, default=5678)
    parser.add_argument("--seed", type=int, default=42)

    # train config
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--total_steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint_interval", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--pipeline", type=str, default="PromptPipeline")
    parser.add_argument("--trainer", type=str, default="AcceleratePPOTrainer")
    parser.add_argument("--num_layers_unfrozen", type=int, default=2)
    parser.add_argument("--save_best", action="store_true")
    parser.add_argument("--seq_length", type=int, default=1024)

    # optimizer config
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.95])
    parser.add_argument("--eps", type=float, default=1.0e-8)
    parser.add_argument("--weight_decay", type=float, default=1.0e-6)
    parser.add_argument("--T_max", type=int, default=10000)
    parser.add_argument("--eta_min", type=float, default=1.0e-5)

    # PPO config
    parser.add_argument("--num_rollouts", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--init_kl_coef", type=float, default=0.05)
    parser.add_argument("--target", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--cliprange_value", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=1)
    parser.add_argument("--scale_reward", type=str, default="ignored")
    parser.add_argument("--ref_mean", type=float, default=None)
    parser.add_argument("--ref_std", type=float, default=None)
    parser.add_argument("--cliprange_reward", type=float, default=10)

    # gen kwargs
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    slurm_job_id = int(os.getenv("SLURM_ARRAY_JOB_ID", 0))
    slurm_array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
    if not "WANDB_RUN_ID" in os.environ:
        wandb_group = f"{args.model}_job_{slurm_job_id}"
        os.environ["WANDB_RUN_GROUP"] = wandb_group

    if args.debug and is_main_process():
        print(f"Attaching debugger to main process with PID {os.getpid()}: {args.debug_port}")
        attach_debugger(args.debug_port)

    main(args)
