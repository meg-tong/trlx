import subprocess
import argparse
from pathlib import Path

cur_dir = Path(__file__).parent
path_to_trlx = cur_dir.parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--config_file", type=str, default="trlx/configs/accelerate/zero2-bf16.yaml")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--partition", type=str, default="compute", choices=["compute", "interactive"])
    parser.add_argument("--experiment_file", type=str, default=str(path_to_trlx / "examples" / "ppo_sentiments_llama.py"))

    args, extra = parser.parse_known_args()

    command = [
        "sbatch", 
        "--gpus-per-node", str(args.num_gpus),
        "--array", f"0-{args.num_runs-1}", 
        "--partition", args.partition,
        str(cur_dir / "slurm_train_deepspeed.sh"),
        # python
        str(cur_dir / "accelerate_launch_wrapper.py"),
        "--num_processes", str(args.num_gpus),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps), 
        "--config_file", args.config_file,
        "--experiment_file", args.experiment_file,
        *extra,
    ]

    print('\n> [train.py] Running: ', " ".join(command), '\n')

    subprocess.run(command)
