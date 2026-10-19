import os
import json
import random
import argparse
import re
from memory_bank import sanitize_memory_text

import openai
openai.api_key = os.environ["OPENAI_API_KEY"]
from openai import OpenAI
client = OpenAI()

# %% load examples
def load_blocks(path: str) -> list[list[str]]:
    """Load blank-line separated blocks from the log file."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace(
        'browsergym.experiments.loop - INFO - action:', '\n\naction:')
    blocks, block = [], []
    for line in content.splitlines():
        if not line.strip():
            if block:
                blocks.append(block)
                block = []
        else:
            block.append(line.strip())
    if block:
        blocks.append(block)

    valid_blocks = []
    for block in blocks:
        is_valid = False
        for line in block:
            if line.startswith('action:'):
                is_valid = True
                break
            if 'browsergym.experiments.loop' in line and 'Running experiment' not in line:
                is_valid = True
                break
        if is_valid:
            valid_blocks.append(block)
    
    assert len(valid_blocks) % 2 == 0
    return valid_blocks

def remove_invalid_steps(actions: list[str]) -> list[str]:
    """Remove invalid steps from the action sequence."""
    valid_actions = []
    for a in actions:
        if "click(" in a:
            arg = a[a.index("(")+1: a.index(")")]
            # if type(eval(arg)) == str:
            if type(arg) == str:
                valid_actions.append(a)
        elif "fill(" in a:
            arg = a[a.index("(")+1: a.index(",")].strip()
            # if type(eval(arg)) == str:
            if type(arg) == str:
                valid_actions.append(a)
        else:
            valid_actions.append(a)
    return valid_actions

def extract_think_and_action(path: str) -> tuple[list[str], list[str]]:
    """Extract the task trajectory from the log file."""
    blocks = load_blocks(path)
    think_list, action_list = [], []
    for i in range(1, len(blocks), 2):
        # action
        b = blocks[i]
        # print(f"Here is action block:\n{b}")
        actions = remove_invalid_steps(b[1:])
        # print(f"Here is actions:\n{actions}")
        if len(actions) == 0: continue
        action_list.append(actions)
        # think
        b = blocks[i-1]
        # print(f"Here is think block:\n{b}")
        think_line = b[-1]
        for line in b:
            if 'browsergym.experiments.loop' in line:
                think_line = line
        idx = think_line.index("browsergym.experiments.loop - INFO -")
        # print(f"Here is think :\n{think_line[idx+36: ].strip()}")
        think_list.append(think_line[idx+36: ].strip())
    
    assert len(think_list) == len(action_list)
    
    # TODO: merge same actions
    return think_list, action_list

def format_trajectory(think_list: list[str], action_list: list[list[str]]) -> str:
    trajectory = []
    for t, a in zip(think_list, action_list):
        acts = '\n'.join(a)
        trajectory.append(f"<think>\n{t}\n</think>\n<action>\n{acts}\n</action>")
    return '\n\n'.join(trajectory)

def random_group_sample(d: dict, n) -> list:
    """Randomly sample n groups from the dictionary."""
    return [ex for v in d.values() for ex in random.sample(v, min(n, len(v)))]

# %% prompt model
# def format_examples(examples: list[dict]) -> str:
#     """Format examples to the prompt."""
#     formatted_examples = []
#     for ex in examples:
#         trajectory = format_trajectory(ex["think_list"], ex["action_list"])
#         formatted_examples.append(f"Query: {ex['query']}\nActions:\n{trajectory}")
#     return '\n\n'.join(["## Concrete Examples"] + formatted_examples + ["## Summary Workflows"])

def llm_generate(examples: dict, args, verbose: bool = False, is_success: bool = True):
    trajectory = format_trajectory(examples["think_list"], examples["action_list"])
    input_prompt = f"Query: {examples['query']}\n\nTrajectory:\n{trajectory}"

    """Call gpt model to generate memory_item."""
    if is_success:
        prompt = '\n\n'.join([args.SUCCEED_PROMPT, input_prompt])
    else:
        prompt = '\n\n'.join([args.FAILED_PROMPT, input_prompt])
    if verbose: print("Prompt:\n", prompt, '\n\n')

    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.reflection_temperature,
        max_tokens=2048,
    )
    response = response.choices[0].message.content
    
    # 后处理：移除 <think> 标签及其内容
    response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
    response = sanitize_memory_text(response)
    if not response:
        raise ValueError("Memory induction returned empty text")

    if verbose: print(response)
    return response, trajectory


def main():
    record_path = os.path.join(args.result_dir, f"webarena.{args.task_id}", "gpt-3.5-turbo_autoeval.json")
    log_path = os.path.join(args.result_dir, f"webarena.{args.task_id}", "experiment.log")
    with open(record_path, encoding="utf-8") as f:
        record = json.load(f)
    if (not isinstance(record, list) or len(record) != 1
            or not isinstance(record[0], dict) or type(record[0].get("rm")) is not bool):
        raise ValueError("Memory induction requires one valid Boolean task-completion rating")
    is_success = record[0]["rm"]
    if not args.bank_path:
        raise ValueError("--bank_path is required for memory induction")

    existing_data = []
    if os.path.exists(args.bank_path):
        with open(args.bank_path, encoding="utf-8") as f:
            existing_data = json.load(f)
        if not isinstance(existing_data, list) or not all(isinstance(item, dict) for item in existing_data):
            raise ValueError("Memory bank must be a list of task records")

    config_path = os.path.join("config_files", f"{args.task_id}.json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    think_list, action_list = extract_think_and_action(log_path)
    wdict = {"query": config["intent"], "think_list": think_list, "action_list": action_list}
    memory_item, trajectory = llm_generate(wdict, args, False, is_success)
    bank_dict = {
        "task_id": args.task_id, "task_idx": int(args.task_id), "sites": config["sites"],
        "query": config["intent"], "success": is_success,
        "trajectory": trajectory, "memory_item": memory_item,
    }
    existing_data = [item for item in existing_data
                     if str(item.get("task_id", item.get("task_idx"))) != str(args.task_id)]
    existing_data.append(bank_dict)
    os.makedirs(os.path.dirname(os.path.abspath(args.bank_path)), exist_ok=True)
    temporary = args.bank_path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4)
    os.replace(temporary, args.bank_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="results",
                        help="Path to the result directory. Support multiple directories separated by space.")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", 
                        choices=[
                            "gpt-3.5",
                            "gpt-4",
                            "gpt-4o",
                            "gpt-4o-2024-11-20",
                            "gpt-4o-mini",
                            "gemini-2.5-flash",
                            "gemini-2.5-pro",
                            "qwen3-235b-a22b-instruct-2507",
                            "gemini-3-pro-preview",
                            "deepseek-v3.2",
                            "claude-sonnet-4-5-20250929",
                            "Qwen3-14B",
                            "Qwen3-32B",
                            "Qwen3-8B"
                        ])
    parser.add_argument("--task_id", type=str, default="0", help="Current task id.")
    parser.add_argument("--bank_path", type=str, default=None, help="Path to the bank saving directory.")
    # parser.add_argument("--criteria", type=str, default="autoeval", 
    #                     choices=["gt", "autoeval"],
    #                     help="'gt': only use examples with gold reward, 'autoeval': use examples with autoeval reward.")
    
    # parser.add_argument("--num_samples", type=int, default=1, help="Max number of samples to input per template.")
    parser.add_argument("--website", type=str, required=True,
                        choices=["shopping", "shopping_admin", "gitlab", "reddit", "map", "multi"])
    parser.add_argument('--max_memory_items', type=int, default=3)
    parser.add_argument('--reflection_temperature', type=float, default=1.0)
    args = parser.parse_args()
    if args.max_memory_items < 1:
        parser.error('--max_memory_items must be positive')

    args.SUCCEED_PROMPT = open("prompt/succeed_prompt.txt", 'r').read()
    args.FAILED_PROMPT = open("prompt/failed_prompt.txt", 'r').read()

    for name in ['SUCCEED_PROMPT', 'FAILED_PROMPT']:
        text = getattr(args, name).replace('at most 3', f'at most {args.max_memory_items}')
        setattr(args, name, text)
    main()
