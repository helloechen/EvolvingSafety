import os
import json
import argparse
from subprocess import Popen
import subprocess
from threading import Timer
import shutil
import time
import sys
from pathlib import Path

def get_model_name_with_prefix(model_name):
    """Add appropriate prefix to model name for run.py."""
    # Models routed through the OpenAI-compatible endpoint
    custom_models = {
        "qwen3-235b-a22b-instruct-2507",
        "gemini-3-pro-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "deepseek-v3.2",
    }
    openai_models = {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-4o-2024-11-20",
        "claude-sonnet-4-5-20250929",
        "Qwen3-14B",
        "Qwen3-32B",
        "Qwen3-8B"
    }

    if model_name in custom_models or model_name in openai_models or model_name.startswith("gpt-"):
        return f"openai/{model_name}"

    # Default to Google prefix for the remaining legacy options.
    return f"google/{model_name}"

def boolean_option(value):
    normalized = str(value).lower()
    if normalized not in {"true", "false"}:
        raise argparse.ArgumentTypeError("Expected true or false")
    return normalized.capitalize()


def check_completed_task(record_path, bank_path, task_id):
    """Do not skip a task whose memory update was interrupted."""
    record = json.loads(Path(record_path).read_text())
    if (not isinstance(record, list) or len(record) != 1
            or not isinstance(record[0], dict) or type(record[0].get("rm")) is not bool):
        raise RuntimeError(f"Task {task_id} has invalid completion feedback")
    bank = json.loads(Path(bank_path).read_text()) if Path(bank_path).exists() else []
    if not isinstance(bank, list) or not all(isinstance(item, dict) for item in bank):
        raise RuntimeError("Cannot resume from an invalid memory bank")
    matching = [item for item in bank
                if str(item.get("task_id", item.get("task_idx"))) == str(task_id)]
    if (len(matching) != 1 or not isinstance(matching[0].get("memory_item"), str)
            or not matching[0]["memory_item"].strip()):
        raise RuntimeError(
            f"Task {task_id} has feedback but no unique saved memory. "
            "Restore a matching bank/results checkpoint before resuming."
        )


def record_run_configuration(result_dir, bank_path, options):
    """Keep model and memory settings consistent when reusing outputs."""
    config = {key: value for key, value in vars(options).items()
              if key not in {"start_index", "end_index", "restart"}}
    path = Path(result_dir) / 'run_config.json'
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError("Run configuration differs; use a fresh bank and results directory")
    else:
        if Path(bank_path).exists() or any(Path(result_dir).glob('webarena.*')):
            raise ValueError("Existing bank/results have no configuration record; preserve them and start fresh")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = str(path) + '.tmp'
        Path(temporary).write_text(json.dumps(config, sort_keys=True, indent=2) + '\n')
        os.replace(temporary, path)


def main():
    for option in ("use_bank", "restart", "is_wo", "faithfulness_experiment"):
        setattr(args, option, boolean_option(getattr(args, option)))
    # collect examples
    config_files = [
        os.path.join("config_files", f) for f in os.listdir("config_files")
        if f.endswith(".json") and f.split(".")[0].isdigit()
    ]
    config_files = sorted(config_files, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    config_list = [json.loads(Path(f).read_text()) for f in config_files]
    # 筛选逻辑：支持 multi 选项
    if args.website == "multi":
        # multi 选项：选择所有多站点任务（len(sites) > 1）
        config_flags = [len(config["sites"]) > 1 for config in config_list]
    else:
        # 原有逻辑：单站点任务且第一个站点匹配
        config_flags = [config["sites"][0] == args.website and len(config["sites"]) == 1 for config in config_list]
    task_ids = [config["task_id"] for config, flag in zip(config_list, config_flags) if flag]

    
    result_dir = 'results'
    if args.is_wo == 'True':
        result_dir = 'results_reddit'
    log_dir = 'log'
    workflow_dir = ''
    log_path = 'logs/'
    if args.faithfulness_experiment == 'True':
        if args.fewshot_modification_type == 'shuffle':
            result_dir += '/fewshot/shuffle'
            log_dir += '/fewshot/shuffle'
            workflow_dir += 'fewshot/shuffle'
            log_path += 'fewshot_shuffle_'
        elif args.fewshot_modification_type == 'irrelevant':
            result_dir += '/fewshot/irrelevant'
            log_dir += '/fewshot/irrelevant'
            workflow_dir += 'fewshot/irrelevant'
            log_path += 'fewshot_irrelevant_'
        elif args.insights_modification_type == 'wo':
            result_dir += '/insights/wo'
            log_dir += '/insights/wo'
            workflow_dir += 'insights/wo'
            log_path += 'insights_wo_'
        elif args.insights_modification_type == 'corrupt':
            result_dir += '/insights/corrupt'
            log_dir += '/insights/corrupt'
            workflow_dir += 'insights/corrupt'
            log_path += 'insights_corrupt_'
        elif args.insights_modification_type == 'irrelevant':
            result_dir += '/insights/irrelevant'
            log_dir += '/insights/irrelevant'
            workflow_dir += 'insights/irrelevant'
            log_path += 'insights_irrelevant_'
        elif args.insights_modification_type == 'filler_tokens':
            result_dir += '/insights/filler_tokens'
            log_dir += '/insights/filler_tokens'
            workflow_dir += 'insights/filler_tokens'
            log_path += 'insights_filler_tokens_'
    
    if args.use_bank == 'False':
        result_dir += '/nobank'
        log_dir += '/nobank'
        workflow_dir += 'nobank'
        log_path += 'nobank'
    log_path += args.website + '.txt'
    for directory in (result_dir, log_dir, os.path.dirname(log_path), f'bank/{workflow_dir}'):
        os.makedirs(directory, exist_ok=True)

    record_run_configuration(result_dir, f"bank/{workflow_dir}/{args.website}.json", args)

    index = len(task_ids)
    for i in range(len(task_ids)):
        if task_ids[i] >= args.start_index:
            index = i
            break
    if args.end_index == None: args.end_index = len(task_ids)
    
    for _ in range(2):
        for tid in task_ids[index: args.end_index]:
            record_path = os.path.join(result_dir, f"webarena.{tid}", "gpt-3.5-turbo_autoeval.json")
            if os.path.exists(record_path):
                check_completed_task(record_path, f"bank/{workflow_dir}/{args.website}.json", tid)
                continue
            # step 1: run inference
            if os.path.exists(f'{result_dir}/webarena.{tid}'):
                for attempt in range(3):
                    try:
                        shutil.rmtree(f'{result_dir}/webarena.{tid}')
                    except Exception as e:
                        time.sleep(1)
            process = Popen([
                sys.executable, "run.py", 
                "--task", f"webarena.{tid}",
                "--bank_path", f"bank/{workflow_dir}/{args.website}.json",
                "--retrieve_k", str(args.retrieve_k),
                "--headless", "True",
                "--embeddings_path", args.embeddings_path or "",
                "--temperature", str(args.temperature),
                "--result_dir", f"{result_dir}",
                "--faithfulness_experiment", f"{args.faithfulness_experiment}",
                "--fewshot_modification_type", f"{args.fewshot_modification_type}",
                "--insights_modification_type", f"{args.insights_modification_type}",
                "--log_path", f"{log_path}",
                "--model_name", get_model_name_with_prefix(args.model_name),
                "--use_bank", f"{args.use_bank}"
            ]) 

            try:
                process.wait(timeout=600)
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, "browser rollout")
            except subprocess.TimeoutExpired:
                print("Process timed out, terminating...")
                process.terminate()  # 尝试终止子进程
                process.wait()       # 等待进程结束
                raise
            
            # step 2: run evaluation
            autoeval_env = os.environ.copy()
            process = Popen([
                sys.executable, "-m", "autoeval.evaluate_trajectory",
                "--result_dir", f"{result_dir}/webarena.{tid}",
                "--log_dir", f"{log_dir}",
                "--model", "gpt-3.5-turbo"  # f"{args.model_name}"
            ], env=autoeval_env)
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, "pipeline stage")

            # step 3: update memory
            process = Popen([
                sys.executable, "induce_prompt.py",
                "--result_dir", f"{result_dir}",
                "--max_memory_items", str(args.max_memory_items),
                "--reflection_temperature", str(args.reflection_temperature),
                "--task_id", f"{tid}",
                "--bank_path", f"bank/{workflow_dir}/{args.website}.json",
                "--model", f"{args.model_name}",
                "--website", f"{args.website}"
            ])
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, "pipeline stage")
    
    if args.restart == 'True':
        for i in range(2):
            for tid in task_ids[index: args.end_index]:
                dir = 'config_files/' + f"{tid}" + '.json'
                with open(dir, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 支持 multi 选项
                if args.website == "multi":
                    should_process = len(data['sites']) > 1
                else:
                    should_process = len(data['sites']) == 1 and data['sites'][0] == args.website
                if should_process:
                        if not os.path.exists(f'{result_dir}/webarena.{tid}/gpt-3.5-turbo_autoeval.json'):
                            print(tid)
                            if os.path.exists(f'{result_dir}/webarena.{tid}'):
                                for attempt in range(3):
                                    try:
                                        shutil.rmtree(f'{result_dir}/webarena.{tid}')
                                    except Exception as e:
                                        time.sleep(1)
                            # step 1: run inference
                            process = Popen([
                                sys.executable, "run.py", 
                                "--task", f"webarena.{tid}",
                                "--bank_path", f"bank/{workflow_dir}/{args.website}.json",
                                "--retrieve_k", str(args.retrieve_k),
                                "--headless", "True",
                                "--embeddings_path", args.embeddings_path or "",
                                "--temperature", str(args.temperature),
                                "--result_dir", f"{result_dir}",
                                "--faithfulness_experiment", f"{args.faithfulness_experiment}",
                                "--fewshot_modification_type", f"{args.fewshot_modification_type}",
                                "--insights_modification_type", f"{args.insights_modification_type}",
                                "--log_path", f"{log_path}",
                                "--model_name", get_model_name_with_prefix(args.model_name),
                                "--use_bank", args.use_bank
                            ]) 
                            try:
                                process.wait(timeout=600)
                                if process.returncode != 0:
                                    raise subprocess.CalledProcessError(process.returncode, "browser rollout")
                            except subprocess.TimeoutExpired:
                                print("Process timed out, terminating...")
                                process.terminate()  # 尝试终止子进程
                                process.wait()       # 等待进程结束
                                raise

                            # step 2: run evaluation
                            autoeval_env = os.environ.copy()
                            process = Popen([
                                sys.executable, "-m", "autoeval.evaluate_trajectory",
                                "--result_dir", f"{result_dir}/webarena.{tid}",
                                "--log_dir", f"{log_dir}",
                                "--model", "gpt-3.5-turbo"  # f"{args.model_name}"
                            ], env=autoeval_env)
                            process.wait()
                            if process.returncode != 0:
                                raise subprocess.CalledProcessError(process.returncode, "pipeline stage")

                            # step 3: update memory
                            process = Popen([
                                sys.executable, "induce_prompt.py",
                                "--result_dir", f"{result_dir}",
                                "--max_memory_items", str(args.max_memory_items),
                                "--reflection_temperature", str(args.reflection_temperature),
                                "--task_id", f"{tid}",
                                "--bank_path", f"bank/{workflow_dir}/{args.website}.json",
                                "--model", f"{args.model_name}",
                                "--website", f"{args.website}"
                            ])
                            process.wait()
                            if process.returncode != 0:
                                raise subprocess.CalledProcessError(process.returncode, "pipeline stage")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", type=str, required=True,
                        choices=["shopping", "shopping_admin", "gitlab", "reddit", "map", "multi"])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--model_name", type=str, default='gemini-2.5-flash')
    parser.add_argument("--faithfulness_experiment", type=boolean_option, default='False')
    parser.add_argument("--fewshot_modification_type", type=str, default='None')
    parser.add_argument("--insights_modification_type", type=str, default='None')
    parser.add_argument("--use_bank", type=boolean_option, default='True')
    parser.add_argument("--restart", type=boolean_option, default='False')
    parser.add_argument("--is_wo", type=boolean_option, default='False')
    parser.add_argument('--embeddings_path', default=None)
    parser.add_argument('--retrieve_k', type=int, default=1)
    parser.add_argument('--max_memory_items', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--reflection_temperature', type=float, default=1.0)
    args = parser.parse_args()
    if args.use_bank.lower() == 'true' and (not args.embeddings_path or not os.path.isfile(args.embeddings_path)):
        parser.error('--embeddings_path must name a complete source-task index')
    if args.retrieve_k < 1 or args.max_memory_items < 1:
        parser.error('Memory counts must be positive')
    main()
