import os
import subprocess
import argparse
from pathlib import Path

DEFAULT_MASTER_PORT = 29500

path_to_trlx = Path(__file__).parent.parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--config_file", type=str, default="trlx/configs/accelerate/zero2-fp16.yaml")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--experiment_file", type=str, default=str(path_to_trlx / "examples" / "ppo_sentiments_llama.py"))
    parser.add_argument("--mixed_precision", type=str, default="fp16")

    args, extra = parser.parse_known_args()

    slurm_job_id = int(os.getenv("SLURM_ARRAY_JOB_ID", 0))
    slurm_array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
    seed = "1000"#str(slurm_job_id + slurm_array_task_id)

    command = [
        "accelerate", "launch",
        "--num_processes", str(args.num_processes),
        "--main_process_port", str((DEFAULT_MASTER_PORT + slurm_job_id + slurm_array_task_id) % 65000),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps), 
        "--config_file", args.config_file,
        "--mixed_precision", args.mixed_precision,
        args.experiment_file,
        "--seed", seed,
        *extra,
    ]

    print('\n> [accelerate_launch_wrapper.py] Running: ', " ".join(command), '\n')

    subprocess.run(command)
