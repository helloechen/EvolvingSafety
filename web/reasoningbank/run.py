"""
WARNING DEPRECATED WILL BE REMOVED SOON
"""

import os
import argparse
from pathlib import Path

from browsergym.experiments import ExpArgs, EnvArgs

from agents.legacy.agent import GenericAgentArgs
from agents.legacy.dynamic_prompting import Flags
from agents.legacy.utils.chat_api import ChatModelArgs
import json
from memory_bank import retrieve_memory


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with hyperparameters.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/gpt-4o",
        help="Model name for the chat model.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="openended",
        help="Name of the Browsergym task to run. If 'openended', you need to specify a 'start_url'",
    )
    parser.add_argument(
        "--start_url",
        type=str,
        default="https://www.google.com",
        help="Starting URL (only for the openended task).",
    )
    parser.add_argument(
        "--slow_mo", type=int, default=30, help="Slow motion delay for the playwright actions."
    )
    parser.add_argument(
        "--headless",
        type=str2bool,
        default=False,
        help="Run the experiment in headless mode (hides the browser windows).",
    )
    parser.add_argument(
        "--demo_mode",
        type=str2bool,
        default=True,
        help="Add visual effects when the agents performs actions.",
    )
    parser.add_argument(
        "--use_html", type=str2bool, default=False, help="Use HTML in the agent's observation space."
    )
    parser.add_argument(
        "--use_ax_tree",
        type=str2bool,
        default=True,
        help="Use AX tree in the agent's observation space.",
    )
    parser.add_argument(
        "--use_screenshot",
        type=str2bool,
        default=True,
        help="Use screenshot in the agent's observation space.",
    )
    parser.add_argument(
        "--multi_actions", type=str2bool, default=True, help="Allow multi-actions in the agent."
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="bid",
        choices=["python", "bid", "coord", "bid+coord", "bid+nav", "coord+nav", "bid+coord+nav"],
        help="",
    )
    parser.add_argument(
        "--use_history",
        type=str2bool,
        default=True,
        help="Use history in the agent's observation space.",
    )
    parser.add_argument(
        "--use_thinking",
        type=str2bool,
        default=True,
        help="Use thinking in the agent (chain-of-thought prompting).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10,
        help="Maximum number of steps to take for each task.",
    )
    parser.add_argument(
        "--bank_path",
        type=str,
        default=None,
        help="Path to the memory file to load for the agent.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default='results',
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default='logs/log.txt',
    )

    parser.add_argument(
        "--faithfulness_experiment",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--fewshot_modification_type",
        type=str,
        default='None',
    )
    parser.add_argument(
        "--insights_modification_type",
        type=str,
        default='None',
    )
    parser.add_argument(
        "--use_bank",
        type=str2bool,
        default=True,
    )

    parser.add_argument('--embeddings_path', default=None)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--retrieve_k', type=int, default=1,
                        help='Number of source-task memory bundles to retrieve')
    args = parser.parse_args()
    if args.use_bank and not args.bank_path:
        parser.error('--bank_path is required when --use_bank is true')
    if args.use_bank and (not args.embeddings_path or not Path(args.embeddings_path).is_file()):
        parser.error('--embeddings_path must name a complete source-task index')
    if args.retrieve_k < 1:
        parser.error('--retrieve_k must be positive')
    return args


def retrieve_for_task(args, tid):
    return retrieve_memory(
        args.bank_path, args.embeddings_path,
        query_id=tid, top_k=args.retrieve_k,
    )


def main():
    args = parse_args()
    if args.use_bank:
        bank_path = Path(args.bank_path)
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        if not bank_path.exists():
            bank_path.write_text("[]\n", encoding="utf-8")

    env_args = EnvArgs(
        task_name=args.task_name,
        task_seed=None,
        max_steps=args.max_steps,
        headless=args.headless,
        viewport={"width": 1500, "height": 1280},
        slow_mo=args.slow_mo,
    )

    if args.task_name == "openended":
        env_args.wait_for_user_message = True
        env_args.task_kwargs = {"start_url": args.start_url}
    
    tid = int(args.task_name.split('.')[1])
    retrieval_metadata = {"selected_task_ids": [], "enabled": args.use_bank}
    if args.use_bank:
        instruction = 'Below are some memory items that I accumulated from past interaction from the environment that may be helpful to solve the task. You can use it when you feel it’s relevant. In each step, please first explicitly discuss if you want to use each memory item or not, and then take action.'
        memory_item, retrieval_metadata = retrieve_for_task(args, tid)
        if memory_item:
            memory_item = instruction + '\n\n' + memory_item
        else:
            memory_item = ''
    else:
        memory_item = ''
    
    # print('-' * 50)
    # print(f"Here is memory_item:\n{memory_item}\n")
    # print('-' * 50)

    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=GenericAgentArgs(
            chat_model_args=ChatModelArgs(
                model_name=args.model_name,
                max_total_tokens=128_000,  # "Maximum total tokens for the chat model."
                max_input_tokens=126_000,  # "Maximum tokens for the input to the chat model."
                max_new_tokens=2_000,  # "Maximum total tokens for the chat model."
                temperature=args.temperature,
            ),
            flags=Flags(
                use_html=args.use_html,
                use_ax_tree=args.use_ax_tree,
                use_thinking=args.use_thinking,  # "Enable the agent with a memory (scratchpad)."
                use_error_logs=True,  # "Prompt the agent with the error logs."
                use_memory=False,  # "Enables the agent with a memory (scratchpad)."
                use_history=args.use_history,
                use_diff=False,  # "Prompt the agent with the difference between the current and past observation."
                use_past_error_logs=True,  # "Prompt the agent with the past error logs."
                use_action_history=True,  # "Prompt the agent with the action history."
                multi_actions=args.multi_actions,
                use_abstract_example=True,  # "Prompt the agent with an abstract example."
                use_concrete_example=True,  # "Prompt the agent with a concrete example."
                use_screenshot=args.use_screenshot,
                enable_chat=True,
                demo_mode="default" if args.demo_mode else "off",
                bank_path=args.bank_path,
                memory_item=memory_item
            ),
            faithfulness_experiment=args.faithfulness_experiment,
            fewshot_modification_type=args.fewshot_modification_type,
            insights_modification_type=args.insights_modification_type,
            log_path = args.log_path
        ),
    )

    exp_args.prepare(Path(f"./{args.result_dir}"))
    (Path(exp_args.exp_dir) / "memory_retrieval.json").write_text(
        json.dumps(retrieval_metadata, indent=2) + "\n", encoding="utf-8"
    )
    exp_args.run()

    os.rename(exp_args.exp_dir, f"{args.result_dir}/{args.task_name}")


if __name__ == "__main__":
    main()
