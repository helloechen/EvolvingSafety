from subprocess import Popen
import subprocess
from argparse import ArgumentParser
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen3-4B")
parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
parser.add_argument('--use_bank', type=str2bool, default=True)
parser.add_argument('--experiment', type=str2bool, default=True)
parser.add_argument('--bank_count', type=int, default=0)
parser.add_argument('--long_sys', type=str2bool, default=False)
parser.add_argument('--workflow', type=str2bool, default=False)
parser.add_argument('--retrival_num', type=int, default=3, choices=[1, 3, 9])
parser.add_argument('--task_timeout', type=int, default=120, help="Per-task timeout in seconds; 0 disables timeout")
parser.add_argument("--safety_mode", type=str, default="normal")
parser.add_argument('--base_count', type=int, default=10)
args = parser.parse_args()

for count in [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 269]:
    for _ in range(3):
        process = Popen([
            "python", "-u", "safety/pipeline_detailed.py", 
            "--model_name", f"{args.model_name}",
            "--retrival_num", f"{args.retrival_num}",
            "--use_bank", f"{args.use_bank}",
            "--experiment", f"{args.experiment}",
            "--bank_count", f"{count}",
            "--safety_mode", f"{args.safety_mode}",
            "--workers", f"{args.workers}",
            "--base_count", f"{args.base_count}"
        ]) 

        try:
            process.wait(timeout=3000)
            break
        except subprocess.TimeoutExpired:
            print("Process timed out, terminating...")
            process.terminate()  # 尝试终止子进程
            process.wait()       # 等待进程结束

# for mode in ['mix', 'combine', 'combine2']:
#     for _ in range(3):
#         process = Popen([
#             "python", "-u", "safety/pipeline_detailed.py", 
#             "--model_name", f"{args.model_name}",
#             "--retrival_num", f"{args.retrival_num}",
#             "--use_bank", f"{args.use_bank}",
#             "--experiment", f"{args.experiment}",
#             "--bank_count", f"0",
#             "--safety_mode", f"{mode}"
#         ]) 

#         try:
#             process.wait(timeout=3000)
#             break
#         except subprocess.TimeoutExpired:
#             print("Process timed out, terminating...")
#             process.terminate()  # 尝试终止子进程
#             process.wait()       # 等待进程结束