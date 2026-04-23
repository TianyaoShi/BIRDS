import argparse
import asyncio
import base64
import io
import json
import os
import random
import time
import warnings
import numpy as np
import pandas as pd
from itertools import cycle
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Collection, Dict, List, Optional, Tuple



from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)


from datasets import load_dataset
from PIL.Image import Image
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from vllm.tokenizers import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

from vllm.v1.structured_output.backend_xgrammar import (
    has_xgrammar_unsupported_json_features,
)

from gpu_monitor import GPUMonitor


POWER_STAT_WARM_UP_INTERVAL = 5


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: List[Tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: List[Tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: List[Tuple[float, float]]


def parse_arguments(args=None):
    print(f"[DEBUG {time.time():.3f}] Starting parse_arguments")
    # see `add_common_args` for extended list of arguments
    parser = argparse.ArgumentParser()
    print(f"[DEBUG {time.time():.3f}] ArgumentParser created")
    parser.add_argument(
        '--draft_engine_dir',
        type=str,
        default=None,
        help='Path to engine of draft model in Draft-Target-Model mode.')
    
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the ShareGPT dataset, will be deprecated in the "
        "next release.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "sonnet", "random", "hf", "longbench", "humaneval", "newsqa"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the sharegpt/sonnet dataset. "
                        "Or the huggingface dataset ID if using HF dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        help="Name or path of the tokenizer, if not using the default tokenizer.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=None,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=float,
        default=1000.0,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=("Number of logprobs-per-token to compute & return as part of "
              "the request. If unspecified, then either (1) if beam search "
              "is disabled, no logprobs are computed & a single dummy "
              "logprob is returned for each token; or (2) if beam search "
              "is enabled 1 logprob per token is computed"),
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl,e2el",
        help="Comma-seperated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        "Allowed metric names are \"ttft\", \"tpot\", \"itl\", \"e2el\". "
        "Default value is \"ttft,tpot,itl\".")
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-seperated list of percentiles for selected metrics. "
        "To report 25-th, 50-th, and 75-th percentiles, use \"25,50,75\". "
        "Default value is \"99\". "
        "Use \"--percentile-metrics\" to select metrics.",
    )
    parser.add_argument(
        "--benchmark-timeout",
        type=int,
        default=600,
        help="Maximum time in seconds for the benchmark to run. "
        "If exceeded, saves partial results and exits gracefully. "
        "Default is 600 seconds (10 minutes).",
    )

    # group for dataset specific arguments
    sonnet_group = parser.add_argument_group("sonnet dataset options")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help=
        "Number of input tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help=
        "Number of output tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help=
        "Number of prefix tokens per request, used only for sonnet dataset.",
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.")
    sharegpt_group.add_argument(
        "--sharegpt-input-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.")
    
    longbench_group = parser.add_argument_group("longbench dataset options")
    longbench_group.add_argument(
        "--longbench-input-len",
        type=int,
        default=None,
        help=
        "Number of input tokens per request, overrides the input lengths "
        "from the sampled longbench dataset.",
    )
    longbench_group.add_argument(
        "--longbench-output-len",
        type=int,
        default=None,
        help=
        "Number of output tokens per request, overrides the output lengths "
        "from the sampled longbench dataset.",
    )

    humaneval_group = parser.add_argument_group("humaneval dataset options")
    humaneval_group.add_argument(
        "--humaneval-input-len",
        type=int,
        default=None,
        help=
        "Number of input tokens per request, used only for humaneval dataset.",
    )
    humaneval_group.add_argument(
        "--humaneval-output-len",
        type=int,
        default=None,
        help=
        "Number of output tokens per request, used only for humaneval dataset.",
    )

    newsqa_group = parser.add_argument_group("newsqa dataset options")
    newsqa_group.add_argument(
        "--newsqa-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the NewsQA dataset.")
    newsqa_group.add_argument(
        "--newsqa-input-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the NewsQA dataset."
    )

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help=
        "Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help=
        "Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random sampling.",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens before random "
        " context. The length range of context in a random "
        " request is [random-prefix-len, "
        " random-prefix-len + random-prefix-len * random-range-ratio).")

    hf_group = parser.add_argument_group("hf dataset options")
    hf_group.add_argument("--hf-subset",
                          type=str,
                          default=None,
                          help="Subset of the HF dataset.")
    hf_group.add_argument("--hf-split",
                          type=str,
                          default=None,
                          help="Split of the HF dataset.")
    hf_group.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output lengths "
        "from the sampled HF dataset.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=lambda s: [int(gpu) for gpu in s.split()],
        default=None,
        help="GPU ids to monitor power consumption (space-separated).",
    )
    parser.add_argument(
        "--gpu-monitor-interval",
        type=float,
        default=0.025,
        help="Interval to monitor GPU power consumption.",
    )
    parser.add_argument(
        "--gpu-monitor-truncate",
        type=float,
        default=0,
        help="Truncate the first and last `truncate` seconds of the monitoring data.",
    )
    parser.add_argument(
        "--use-deterministic-rate",
        action="store_true",
        help="Use deterministic rate instead of exponential distribution.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0,
        help="Ratio of the additional warm up requests to the total number of requests.",
    )

    print(f"[DEBUG {time.time():.3f}] About to call add_common_args")
    # parser = add_common_args(parser)
    print(f"[DEBUG {time.time():.3f}] add_common_args completed")

    print(f"[DEBUG {time.time():.3f}] About to parse arguments")
    result = parser.parse_args(args=args)
    print(f"[DEBUG {time.time():.3f}] Arguments parsed successfully")
    return result


def sample_longbench_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_input_len: Optional[int] = None,
    fixed_output_len: Optional[int] = None,
    max_context_len: Optional[int] = 2048,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)

    base_prompt = "Summarize the following text:\n"
    dataset = [(data['context'],data['answers'] if isinstance(data['answers'],str) else data['answers'][0]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for data in cycle(dataset):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = base_prompt + data[0]
        prompt_token_ids = tokenizer(prompt).input_ids
        if fixed_input_len is not None:
            if len(prompt_token_ids) >= fixed_input_len:
                prompt = tokenizer.decode(prompt_token_ids[:fixed_input_len])
                prompt_token_ids = prompt_token_ids[:fixed_input_len]
            else:
                continue
        completion = data[1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len + output_len > max_context_len:
            # Prune too long sequences.
            continue
        
        filtered_dataset.append((prompt, prompt_len, output_len, None))

    return filtered_dataset

def sample_humaneval_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_input_len: Optional[int] = None,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    # with open(dataset_path) as f:
    #     dataset = json.load(f)
    # dataset = [(data["prompt"], data["canonical_solution"], data["input_tokens"], data["output_tokens"]) for data in dataset]
    df = pd.read_parquet(dataset_path)
    dataset = [(row["prompt"], row["canonical_solution"]) for _, row in df.iterrows()]

    # Shuffle the dataset.
    base_prompt = "Write a python program to complete the following code:\n"
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for data in cycle(dataset):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = base_prompt + data[0]
        # Try to use chat template if available, otherwise just use the prompt directly
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except (ValueError, AttributeError):
            # Fallback for models without chat templates (like llama-1-30b)
            text = prompt
        
        prompt_token_ids = tokenizer(text).input_ids
        if fixed_input_len is not None:
            if len(prompt_token_ids) >= fixed_input_len:
                prompt = tokenizer.decode(prompt_token_ids[:fixed_input_len])
                prompt_token_ids = prompt_token_ids[:fixed_input_len]
            else:
                continue
        completion = data[1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((text, prompt_len, output_len, None))

    return filtered_dataset

def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_input_len: Optional[int] = None,
    fixed_output_len: Optional[int] = None,
    max_context_len: Optional[int] = 4096,
) -> List[Tuple[str, int, int, None]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        # Try to use chat template if available, otherwise just use the prompt directly
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except (ValueError, AttributeError):
            # Fallback for models without chat templates (like llama-1-30b)
            text = prompt
            max_context_len = 2048
        
        prompt_token_ids = tokenizer(text).input_ids
        if fixed_input_len is not None:
            if len(prompt_token_ids) >= fixed_input_len:
                prompt = tokenizer.decode(prompt_token_ids[:fixed_input_len])
                prompt_token_ids = prompt_token_ids[:fixed_input_len]
            else:
                continue
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        

        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue

        if prompt_len > 2048 or prompt_len + output_len > max_context_len:
            # Prune too long sequences.
            continue
        filtered_dataset.append((text, prompt_len, output_len, None))

    return filtered_dataset

def sample_sonnet_requests(
    dataset_path: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    prefix_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, str, int, int, None]]:
    assert (
        input_len > prefix_len
    ), "'args.sonnet-input-len' must be greater than 'args.prefix-input-len'."

    # Load the dataset.
    with open(dataset_path) as f:
        poem_lines = f.readlines()

    # Tokenize the poem lines.
    poem_token_ids = tokenizer(''.join(poem_lines)).input_ids
    # print(poem_token_ids)
    # average_poem_len = sum(
    #     len(token_ids) for token_ids in poem_token_ids) / len(poem_token_ids)
    # FIXME The original implemenation will inflate the average poem length by about 1,
    # which will lead to the result that when you specify --input-len 2048, the actual input length
    # is only ~1900. This is because the tokenizer will add special tokens to the input_ids for each line.
    # But when you concatenate all the lines together, the special tokens will only appear once.
    # Inflated average poem length means less lines will be sampled in each request.
    average_poem_len = len(poem_token_ids) / len(poem_lines)
    # print("Average_poem_len", average_poem_len)

    # Base prefix for all requests.
    base_prompt = "Pick as many lines as you can from these poem lines:\n"
    base_message = [{
        "role": "user",
        "content": base_prompt,
    }]
    base_prompt_formatted = tokenizer.apply_chat_template(
        base_message, add_generation_prompt=True, tokenize=False)
    base_prompt_offset = len(tokenizer(base_prompt_formatted).input_ids)

    assert (
        input_len > base_prompt_offset
    ), f"Please set 'args.sonnet-input-len' higher than {base_prompt_offset}."
    num_input_lines = round(
        (input_len - base_prompt_offset) / average_poem_len)

    # First approximately `prefix_len` number of tokens in the
    # prompt are fixed poem lines.
    assert (
        prefix_len > base_prompt_offset
    ), f"Please set 'args.sonnet-prefix-len' higher than {base_prompt_offset}."

    num_prefix_lines = round(
        (prefix_len - base_prompt_offset) / average_poem_len)
    prefix_lines = poem_lines[:num_prefix_lines]

    # Sample the rest of lines per request.
    sampled_requests: List[Tuple[str, int, int]] = []
    for _ in range(num_requests):
        sampled_lines = "".join(
            prefix_lines +
            random.sample(poem_lines, num_input_lines - num_prefix_lines))

        prompt = f"{base_prompt}{sampled_lines}"
        message = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        prompt_formatted = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False)
        prompt_len = len(tokenizer(prompt_formatted).input_ids)
        sampled_requests.append(
            (prompt, prompt_formatted, prompt_len, output_len, None))

    return sampled_requests


def sample_hf_requests(
    dataset_path: str,
    dataset_subset: str,
    dataset_split: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, str, int, Optional[Dict[str, Collection[str]]]]]:
    dataset = load_dataset(dataset_path,
                           name=dataset_subset,
                           split=dataset_split,
                           streaming=True)
    assert "conversations" in dataset.features, (
        "HF Dataset must have 'conversations' column.")
    filtered_dataset = dataset.shuffle().filter(
        lambda x: len(x["conversations"]) >= 2)
    sampled_requests: List[Tuple[str, int, int, Dict[str,
                                                     Collection[str]]]] = []
    for data in filtered_dataset:
        if len(sampled_requests) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = data["conversations"][0]["value"]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = data["conversations"][1]["value"]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue

        if "image" in data and isinstance(data["image"], Image):
            image: Image = data["image"]
            image = image.convert("RGB")
            image_data = io.BytesIO()
            image.save(image_data, format='JPEG')
            image_base64 = base64.b64encode(
                image_data.getvalue()).decode("utf-8")
            mm_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                },
            }
        else:
            mm_content = None

        sampled_requests.append((prompt, prompt_len, output_len, mm_content))

    return sampled_requests


def sample_random_requests(
    prefix_len: int,
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    prefix_token_ids = np.random.randint(0,
                                         tokenizer.vocab_size,
                                         size=prefix_len).tolist()

    input_lens = np.random.randint(
        int(input_len * range_ratio),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )
    offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
    input_requests = []
    for i in range(num_prompts):
        prompt = tokenizer.decode(prefix_token_ids +
                                  [(offsets[i] + i + j) % tokenizer.vocab_size
                                   for j in range(input_lens[i])])

        input_requests.append((prompt, int(prefix_len + input_lens[i]),
                               int(output_lens[i]), None))

    return input_requests

def sample_newsqa_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_input_len: Optional[int] = None,
    fixed_output_len: Optional[int] = None,
    max_context_len: Optional[int] = 4096
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line.strip()) for line in f]
    base_prompt = "Summarize the following text:\n"
    dataset = [(data['story'], data['summary'])  for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for data in cycle(dataset):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions
        prompt = base_prompt + data[0]
        # Try to use chat template if available, otherwise just use the prompt directly
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except (ValueError, AttributeError):
            # Fallback for models without chat templates (like llama-1-30b)
            text = prompt
            max_context_len = 2048

        prompt_token_ids = tokenizer(text).input_ids
        if fixed_input_len is not None:
            if len(prompt_token_ids) >= fixed_input_len:
                prompt = tokenizer.decode(prompt_token_ids[:fixed_input_len])
                prompt_token_ids = prompt_token_ids[:fixed_input_len]
            else:
                continue
        completion = data[1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        output_len = max(output_len, 200)
        if prompt_len + output_len > max_context_len:
            # Prune too long sequences.
            continue
        
        filtered_dataset.append((text, prompt_len, output_len, None))

    return filtered_dataset

async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    use_deterministic_rate: bool = False,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        
        if use_deterministic_rate:
            interval = 1.0 / request_rate
        else:
            # Sample the request interval from the exponential distribution.
            interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[float],
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2els: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note : this may inflate the output token count slightly
            output_len = len(
                tokenizer(outputs[i].generated_text,
                          add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            if output_len > 1:
                tpots.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000)
                            for p in selected_percentiles],
        mean_e2el_ms=np.median(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.mean(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000)
                             for p in selected_percentiles],
    )

    return metrics, actual_output_lens


async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    logprobs: Optional[int],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[str],
    gpu_monitors: Optional[List[GPUMonitor]] = None,
    warmup_ratio: float = 0.1,
    benchmark_timeout: Optional[int] = None,
    use_deterministic_rate: bool = False,
    gpu_monitor_truncate: float = 0,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len, test_mm_content = (
        input_requests[0])
    if backend != "openai-chat" and test_mm_content is not None:
        # multi-modal benchmark is only available on OpenAI Chat backend.
        raise ValueError(
            "Multi-modal content is only supported on 'openai-chat' backend.")
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        logprobs=logprobs,
        best_of=best_of,
        use_beam_search=use_beam_search,
        multi_modal_content=test_mm_content,
    )
    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}")
    else:
        print("Initial test run completed. Starting main benchmark run...")

    if profile:
        print("Starting profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/start_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            best_of=best_of,
            use_beam_search=use_beam_search,
            multi_modal_content=test_mm_content,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler started")

    print(f"Traffic request rate: {request_rate}")
    print("Start warming up...")
    warmup_length = int(len(input_requests) * warmup_ratio)
    warmup_requests = input_requests[:warmup_length].copy()
    if warmup_length == 0:
        end_up_requests = []
    else:
        end_up_requests = input_requests[-warmup_length:].copy()

    pbar = None if disable_tqdm else tqdm(total=len(input_requests)+warmup_length*2)

    tasks: List[asyncio.Task] = []
    warmup_cnt = 0
    timeout_task = None
    benchmark_start_time = None

    # Create an async generator that we can cancel
    async def create_and_run_requests():
        nonlocal benchmark_start_time, timeout_task
        warmup_cnt = 0
        async for request in get_request(warmup_requests + input_requests + end_up_requests, request_rate, use_deterministic_rate=use_deterministic_rate):
            prompt, prompt_len, output_len, mm_content = request
            request_func_input = RequestFuncInput(
                model=model_id,
                prompt=prompt,
                api_url=api_url,
                prompt_len=prompt_len,
                output_len=output_len,
                logprobs=logprobs,
                best_of=best_of,
                use_beam_search=use_beam_search,
                multi_modal_content=mm_content,
            )
            if warmup_cnt == warmup_length:
                print("Warmup completed. Starting benchmark...")
                benchmark_start_time = time.time()
                
                if benchmark_timeout:
                    print(f"🕐 Running benchmark with {benchmark_timeout}s timeout...")

                if gpu_monitors is not None:
                    for gpu_monitor in gpu_monitors:
                        gpu_monitor.start()

                tasks.append(
                    asyncio.create_task(
                        request_func(request_func_input=request_func_input,
                                    pbar=pbar)))
            elif warmup_cnt == warmup_length + len(input_requests) - 1:
                tasks.append(
                    asyncio.create_task(
                        request_func(request_func_input=request_func_input,
                                    pbar=pbar)))
                print("Benchmark (nearly) completed. Starting end up...")
                if gpu_monitors is not None:
                    for gpu_monitor in gpu_monitors:
                        gpu_monitor.stop()
            else:
                tasks.append(
                    asyncio.create_task(
                        request_func(request_func_input=request_func_input,
                                    pbar=pbar)))
            warmup_cnt += 1

    # Create main benchmark task
    main_task = asyncio.create_task(create_and_run_requests())
    
    # Execute with timeout protection
    try:
        if benchmark_timeout:
            # Wait for benchmark start, then start timeout
            while benchmark_start_time is None:
                await asyncio.sleep(0.01)  # Small delay to let benchmark start
                if main_task.done():
                    break
            
            if benchmark_start_time is not None:
                # Calculate remaining timeout
                elapsed = time.time() - benchmark_start_time
                remaining_timeout = max(0, benchmark_timeout - elapsed)
                
                if remaining_timeout > 0:
                    # Create timeout from benchmark start
                    async def timeout_coroutine():
                        await asyncio.sleep(remaining_timeout)
                    
                    timeout_task = asyncio.create_task(timeout_coroutine())
                    
                    # Wait for either main task completion or timeout
                    done, pending = await asyncio.wait(
                        [main_task, timeout_task], 
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    
                    if timeout_task in done:
                        # Timeout occurred
                        timeout_occurred = True
                        print(f"\n⏰ BENCHMARK TIMEOUT after {benchmark_timeout}s")
                    else:
                        # Completed normally
                        timeout_occurred = False
                else:
                    # Already exceeded timeout
                    main_task.cancel()
                    timeout_occurred = True
                    print(f"\n⏰ BENCHMARK TIMEOUT after {benchmark_timeout}s")
            else:
                # No benchmark started, wait for main task
                await main_task
                timeout_occurred = False
        else:
            # No timeout, just wait for completion
            await main_task
            timeout_occurred = False
            
        # Calculate final duration
        if benchmark_start_time is not None:
            benchmark_duration = time.time() - benchmark_start_time
        else:
            benchmark_duration = 0
            
        # Wait for all created tasks to complete or collect partial results
        if tasks:
            if timeout_occurred:
                # Cancel all running tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                
                # Give tasks a moment to cancel
                await asyncio.sleep(0.1)
                
                # Collect results from completed tasks
                outputs = []
                for task in tasks:
                    if task.done() and not task.cancelled():
                        try:
                            result = task.result()
                            outputs.append(result)
                        except Exception as e:
                            # Create a failed output for errored tasks
                            failed_output = RequestFuncOutput(
                                generated_text="",
                                success=False,
                                latency=benchmark_duration,
                                ttft=0.0,
                                itl=[],
                                prompt_len=0,
                                error=f"Task failed: {str(e)}"
                            )
                            outputs.append(failed_output)
                    else:
                        # Create timeout output for incomplete/cancelled tasks
                        timeout_output = RequestFuncOutput(
                            generated_text="",
                            success=False,
                            latency=benchmark_duration,
                            ttft=0.0,
                            itl=[],
                            prompt_len=0,
                            error="Request timed out or cancelled"
                        )
                        outputs.append(timeout_output)

                if gpu_monitors is not None:
                    for gpu_monitor in gpu_monitors:
                        gpu_monitor.stop()
            else:
                # Wait for all tasks to complete normally
                outputs = await asyncio.gather(*tasks, return_exceptions=True)
                # Convert exceptions to failed outputs
                for i, output in enumerate(outputs):
                    if isinstance(output, Exception):
                        outputs[i] = RequestFuncOutput(
                            generated_text="",
                            success=False,
                            latency=benchmark_duration,
                            ttft=0.0,
                            itl=[],
                            prompt_len=0,
                            error=f"Task failed: {str(output)}"
                        )
        else:
            outputs = []
        
        # Extract only the benchmark results (exclude warmup and end-up)
        if outputs and len(outputs) >= warmup_length:
            benchmark_outputs = outputs[warmup_length:warmup_length + len(input_requests)]
            # Pad if we don't have enough outputs
            while len(benchmark_outputs) < len(input_requests):
                timeout_output = RequestFuncOutput(
                    generated_text="",
                    success=False,
                    latency=benchmark_duration,
                    ttft=0.0,
                    itl=[],
                    prompt_len=0,
                    error="Request not created due to timeout"
                )
                benchmark_outputs.append(timeout_output)
            outputs = benchmark_outputs
        else:
            # Not enough outputs, create placeholder outputs
            outputs = []
            for _ in range(len(input_requests)):
                timeout_output = RequestFuncOutput(
                    generated_text="",
                    success=False,
                    latency=benchmark_duration,
                    ttft=0.0,
                    itl=[],
                    prompt_len=0,
                    error="Request not created due to timeout"
                )
                outputs.append(timeout_output)
        
    except Exception as e:
        # Handle any unexpected errors
        timeout_occurred = True
        benchmark_duration = time.time() - benchmark_start_time if benchmark_start_time else 0
        print(f"Error during benchmark execution: {e}")
        
        # Create error outputs for all requests
        outputs = []
        for _ in range(len(input_requests)):
            error_output = RequestFuncOutput(
                generated_text="",
                success=False,
                latency=benchmark_duration,
                ttft=0.0,
                itl=[],
                prompt_len=0,
                error=f"Benchmark error: {str(e)}"
            )
            outputs.append(error_output)
    
    # Print timeout summary
    if timeout_occurred:
        print(f"🔍 Collecting partial results from completed requests...")
        successful_completions = sum(1 for output in outputs if output.success)
        print(f"📊 Partial results: {successful_completions}/{len(input_requests)} requests completed")

    if profile:
        print("Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    

    # for output in outputs:
    #     print(output)

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):",
                                    metrics.total_token_throughput))

    result = {
        "duration": benchmark_duration,
        "start_time": benchmark_start_time,
        "end_time": benchmark_start_time + benchmark_duration,
        "truncated_time": gpu_monitor_truncate,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "e2els": [output.latency for output in outputs],
        "input_prompts": [input_request[0] for input_request in input_requests],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        "timeout_occurred": timeout_occurred,
        "benchmark_timeout": benchmark_timeout,
    }

    if gpu_monitors is not None:
        result["gpu_power_stats"] = {}
        for gpu_monitor in gpu_monitors:
            monitor_snapshot = gpu_monitor.get_snapshot(wait=True, timeout=1.0)
            avg_power = monitor_snapshot.avg_power_mw
            power_stats = monitor_snapshot.power_stats
            power_trace = monitor_snapshot.power_trace_mw
            print("{s:{c}^{n}}".format(s=f'GPU Power Consumption (W) for GPU {gpu_monitor.gpu_id}', n=50, c='-'))
            print("{:<40} {:<10.2f}".format("Average Power (W):", avg_power/1000))
            print("{s:{c}^{n}}".format(s='Power Statistics (W)', n=50, c='-'))
            print("{:<40} {:<10.2f}".format("Min Power (W):", power_stats["min_power"]/1000))
            print("{:<40} {:<10.2f}".format("5th Percentile Power (W):",
                                            power_stats["power_5p"]/1000))
            print("{:<40} {:<10.2f}".format("25th Percentile Power (W):",
                                            power_stats["power_25p"]/1000))
            print("{:<40} {:<10.2f}".format("Median Power (W):",
                                            power_stats["median_power"]/1000))
            print("{:<40} {:<10.2f}".format("75th Percentile Power (W):",
                                            power_stats["power_75p"]/1000))
            print("{:<40} {:<10.2f}".format("95th Percentile Power (W):",
                                            power_stats["power_95p"]/1000))
            print("{:<40} {:<10.2f}".format("Max Power (W):", power_stats["max_power"]/1000))
            print("{:<40} {:<10.2f}".format("Power Standard Deviation (W):",
                                            power_stats["power_std"]))
            result["gpu_power_stats"][gpu_monitor.gpu_id] = {
                "avg_power": avg_power,
                "power_stats": power_stats,
                "power_trace": power_trace,
            }
        result["energy_stats"] = {}
        result["energy_stats"]["total_energy"] = result['duration'] * sum([result["gpu_power_stats"][gpu_monitor.gpu_id]["avg_power"] for gpu_monitor in gpu_monitors]) / 1000
        result["energy_stats"]["energy_per_request"] = result["energy_stats"]["total_energy"] / metrics.completed
        result["energy_stats"]["energy_per_token"] = result["energy_stats"]["total_energy"] / (metrics.total_output + metrics.total_input)
        print("{s:{c}^{n}}".format(s='Energy Consumption (J)', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Total Energy (J):", result["energy_stats"]["total_energy"]))
        print("{:<40} {:<10.8f}".format("Total Energy (kWh):", result["energy_stats"]["total_energy"]/3600/1000))
        print("{:<40} {:<10.2f}".format("Energy per Request (J):", result["energy_stats"]["energy_per_request"]))
        print("{:<40} {:<10.2f}".format("Energy per Token processed (J):", result["energy_stats"]["energy_per_token"]))


    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function print and add statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c='-'))
        print("{:<40} {:<10.2f}".format(
            f"Mean {metric_name} (ms):",
            getattr(metrics, f"mean_{metric_attribute_name}_ms")))
        print("{:<40} {:<10.2f}".format(
            f"Median {metric_name} (ms):",
            getattr(metrics, f"median_{metric_attribute_name}_ms")))
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms")
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms")
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms")
        for p, value in getattr(metrics,
                                f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):",
                                            value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT",
                       "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result


def main(args: argparse.Namespace):

    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)

    def check_existing_results(args: argparse.Namespace) -> bool:
        if not args.save_result or not args.result_filename:
            return False
        
        file_path = args.result_filename
        if args.result_dir:
            file_path = os.path.join(args.result_dir, file_path)
            
        if not os.path.exists(file_path):
            return False
            
        try:
            with open(file_path, 'r') as f:
                existing_result = json.load(f)
                
            # Check if result is complete and matches current args
            if (existing_result.get('num_prompts', 0) == int(args.num_prompts) and
                existing_result.get('request_rate') == (args.request_rate if args.request_rate < float('inf') else 'inf')):
                print(f"Complete results already exist in {file_path}, skipping benchmark")
                return True
                
        except (json.JSONDecodeError, KeyError):
            return False
            
        return False

    if check_existing_results(args):
        return 0

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    args.num_prompts = max(int(args.num_prompts),2)

    if args.gpu_ids is not None:
        gpu_monitors = []
        for gpu_id in args.gpu_ids:
            gpu_monitors.append(GPUMonitor(gpu_id=gpu_id, interval=args.gpu_monitor_interval, truncate=args.gpu_monitor_truncate))
    else:
        gpu_monitors = None


    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"


   
    tokenizer = get_tokenizer(
        tokenizer_id,
        trust_remote_code=args.trust_remote_code,
        tokenizer_mode=args.tokenizer_mode,
    )

    if args.dataset is not None:
        warnings.warn(
            "The '--dataset' argument will be deprecated in the next "
            "release. Please use '--dataset-name' and "
            "'--dataset-path' in the future runs.",
            stacklevel=2)
        print(f"[DEBUG {time.time():.3f}] About to sample sharegpt requests (deprecated path)")
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_input_len=args.sharegpt_input_len,
            fixed_output_len=args.sharegpt_output_len,
        )
        print(f"[DEBUG {time.time():.3f}] Sharegpt requests sampled (deprecated path)")

    elif args.dataset_name == "sharegpt":
        print(f"[DEBUG {time.time():.3f}] About to sample sharegpt requests")
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_input_len=args.sharegpt_input_len,
            fixed_output_len=args.sharegpt_output_len,
        )
    
    elif args.dataset_name == "longbench":
        input_requests = sample_longbench_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_input_len=args.longbench_input_len,
            fixed_output_len=args.longbench_output_len,
        )

    elif args.dataset_name == "humaneval":
        print(f"[DEBUG {time.time():.3f}] About to sample humaneval requests")
        input_requests = sample_humaneval_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_input_len=args.humaneval_input_len,
            fixed_output_len=args.humaneval_output_len,
        )
        print(f"[DEBUG {time.time():.3f}] Humaneval requests sampled")

    elif args.dataset_name == "sonnet":
        # Do not format the prompt, pass to message directly
        if args.backend == "openai-chat":
            input_requests = sample_sonnet_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
            )
            input_requests = [(prompt, prompt_len, output_len, None)
                              for prompt, prompt_formatted, prompt_len,
                              output_len, _ in input_requests]
        else:
            assert (
                tokenizer.chat_template or tokenizer.default_chat_template
            ), "Tokenizer/model must have chat template for sonnet dataset."
            input_requests = sample_sonnet_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
            )
            input_requests = [(prompt_formatted, prompt_len, output_len, None)
                              for prompt, prompt_formatted, prompt_len,
                              output_len, _ in input_requests]

    elif args.dataset_name == "hf":
        input_requests = sample_hf_requests(
            dataset_path=args.dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=args.hf_split,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.hf_output_len,
        )

    elif args.dataset_name == "random":
        print(f"[DEBUG {time.time():.3f}] About to sample random requests")
        input_requests = sample_random_requests(
            prefix_len=args.random_prefix_len,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
        )

    elif args.dataset_name == "newsqa":
        input_requests = sample_newsqa_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_input_len=args.newsqa_input_len,
            fixed_output_len=args.newsqa_output_len,
        )

    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    

    # Run the benchmark with timeout protection covering the entire execution  
    benchmark_result = asyncio.run(benchmark(
        backend=backend,
        api_url=api_url,
        base_url=base_url,
        model_id=model_id,
        tokenizer=tokenizer,
        input_requests=input_requests,
        logprobs=args.logprobs,
        best_of=args.best_of,
        use_beam_search=args.use_beam_search,
        request_rate=args.request_rate,
        disable_tqdm=args.disable_tqdm,
        profile=args.profile,
        selected_percentile_metrics=args.percentile_metrics.split(","),
        selected_percentiles=[
            float(p) for p in args.metric_percentiles.split(",")
        ],
        gpu_monitors=gpu_monitors,
        warmup_ratio=args.warmup_ratio,
        benchmark_timeout=args.benchmark_timeout,
        use_deterministic_rate=args.use_deterministic_rate,
        gpu_monitor_truncate=args.gpu_monitor_truncate,
    ))
    
    # Check if timeout occurred and set appropriate status
    if benchmark_result.get("timeout_occurred", False):
        benchmark_result["benchmark_status"] = "timeout"
        completed_requests = benchmark_result.get("completed", 0)
        total_requests = len(input_requests)
        print(f"\n⏰ Benchmark timed out after {args.benchmark_timeout}s")
        print(f"📊 Partial results: {completed_requests}/{total_requests} requests completed")
        print(f"💾 Saving partial benchmark results...")
    else:
        benchmark_result["benchmark_status"] = "completed"
        print(f"\n✅ Benchmark completed successfully")

    # Save config and results to json
    if args.save_result:
        result_json: Dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["use_beam_search"] = args.use_beam_search
        result_json["num_prompts"] = args.num_prompts

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf")

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = f"{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"  #noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
            os.makedirs(args.result_dir, exist_ok=True)
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)

    # Ensure proper cleanup of GPU monitors to prevent thread/fd leaks
    if gpu_monitors:
        for monitor in gpu_monitors:
            try:
                monitor.stop()
                # Force cleanup of monitor threads and resources
                if hasattr(monitor, 'thread') and monitor.thread:
                    monitor.thread.join(timeout=1.0)  # Wait briefly for clean shutdown
            except Exception as e:
                print(f"Warning: GPU monitor cleanup error: {e}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
