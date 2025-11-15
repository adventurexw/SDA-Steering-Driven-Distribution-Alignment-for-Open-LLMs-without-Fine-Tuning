import os
import re
import shutil
import string
from pathlib import Path
from typing import List, Literal, TypedDict

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)
from transformers.generation.utils import GenerationConfig
from accelerate import (
    init_empty_weights,
    load_checkpoint_and_dispatch,
    infer_auto_device_map,
)
from vllm import LLM, SamplingParams

# ===================== Chat template centralization =====================
# Unified chat template dictionary to avoid duplication across functions.
CHAT_TEMPLATES = {
    "qwen": (
        "{% set system_message = '' %}"
        "{% if messages[0]['role'] == 'system' %}"
            "{% set system_message = messages[0]['content'].strip() %}"
            "{% set messages = messages[1:] %}"
        "{% endif %}"
        "{% for message in messages %}"
            "{% set content = message['content'].strip() %}"
            "{% if loop.index0 == 0 and system_message != '' %}"
                "{% set content = system_message + '\\n\\n' + content %}"
            "{% endif %}"
            "{% if message['role'] == 'user' %}"
                "<|im_start|>user\\n{{ content }}<|im_end|>\\n"
            "{% elif message['role'] == 'assistant' %}"
                "<|im_start|>assistant\\n{{ content }}<|im_end|>\\n"
            "{% endif %}"
        "{% endfor %}"
        "<|im_start|>assistant\\n"
    ),
    "deepseek": (
        "{% set system_message = '' %}"
        "{% if messages[0]['role'] == 'system' %}"
            "{% set system_message = messages[0]['content'].strip() %}"
            "{% set messages = messages[1:] %}"
        "{% endif %}"
        "{% for message in messages %}"
            "{% set content = message['content'].strip() %}"
            "{% if loop.index0 == 0 and system_message != '' %}"
                "{% set content = system_message + '\\n\\n' + content %}"
            "{% endif %}"
            "{% if message['role'] == 'user' %}"
                "<|im_start|>user\\n{{ content }}<|im_end|>\\n"
            "{% elif message['role'] == 'assistant' %}"
                "<|im_start|>assistant\\n{{ content }}<|im_end|>\\n"
            "{% endif %}"
        "{% endfor %}"
        "<|im_start|>assistant\\n"
    ),
    "llama-3": (
        "{{ bos_token }}"
        "{% set system_message = '' %}"
        "{% if messages[0]['role'] == 'system' %}"
            "{% set system_message = messages[0]['content'].strip() + '\\n\\n' %}"
            "{% set messages = messages[1:] %}"
        "{% endif %}"
        "{% for message in messages %}"
            "{% set content = message['content'].strip() %}"
            "{% if loop.index0 == 0 and system_message != '' %}"
                "{% set content = system_message + content %}"
            "{% endif %}"
            "<|start_header_id|>{{ message['role'] }}<|end_header_id|>{{ content }}<|eot_id|>"
        "{% endfor %}"
    ),
    "llama-2": (
        "{% set system_message = '' %}"
        "{% if messages and messages[0]['role'] == 'system' %}"
            "{% set system_message = '<<SYS>>\\n' + messages[0]['content'] | trim + '\\n<</SYS>>\\n\\n' %}"
            "{% set messages = messages[1:] %}"
        "{% endif %}"
        "{% if messages %}"
            "{{ bos_token }}"
            "{% for message in messages %}"
                "{% set is_user = message['role'] == 'user' %}"
                "{% set content = message['content'] | trim | default('') %}"
                "{% if loop.index0 == 0 %}"
                    "{% set content = system_message + content %}"
                "{% endif %}"
                "{% if is_user %}"
                    "[INST] {{ content }} [/INST]"
                "{% else %}"
                    "{{ content }}{{ eos_token }}"
                "{% endif %}"
            "{% endfor %}"
        "{% endif %}"
    ),
    "vicuna": (
        "{% for msg in messages %}"
            "{% if msg.role == 'user' %}"
                "USER: {{ msg.content | trim }}"
            "{% else %}"
                "ASSISTANT: {{ msg.content | trim }}"
            "{% endif %}"
            "{% if not loop.last %}\\n{% endif %}"
        "{% endfor %}"
        "{% if messages and messages[-1].role == 'user' %}ASSISTANT: {% endif %}"
    ),
}

def set_chat_template(tokenizer, model_path: str):
    """
    Assign the correct chat template based on model name.
    Raises ValueError if unsupported.
    """
    lower_name = str(model_path).lower()
    for key, template in CHAT_TEMPLATES.items():
        if key in lower_name:
            tokenizer.chat_template = template
            print(f"Chat template set for type: {key}")
            return
    raise ValueError(f"Unsupported model type: {model_path}. Please set tokenizer.chat_template manually.")


# ===================== Small Modules =====================

def load_tokenizer_and_model(
    model: str,
    no_half: bool = False,
    flash_attention: bool = False,
    device_map=None,
    gpu: list[int] = [0],
    mem_limit: int = None,
    no_split: list[str] = []
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load tokenizer and model with explicit parameters."""
    name_or_path = Path(model)
    if not name_or_path.exists():
        ans = input(f'Model "{name_or_path}" not found locally, continue? [y/N]: ')
        if ans.strip().lower() != "y":
            exit(0)

    import warnings
    warnings.filterwarnings(
        "ignore",
        message="A decoder-only architecture is being used, but right-padding was detected!"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        padding_side="left",
        trust_remote_code=True
    )

    dtype = torch.float32 if no_half else torch.bfloat16
    attn = "flash_attention_2" if flash_attention else None

    if device_map is None:
        model_obj = AutoModelForCausalLM.from_pretrained(name_or_path, trust_remote_code=True)
        if not no_half:
            model_obj = model_obj.half()
        model_obj = model_obj.to(gpu[0])
    elif isinstance(device_map, str):
        print(f"Using device map: {device_map} with gpu {os.environ.get('CUDA_VISIBLE_DEVICES','')}")
        model_obj = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            trust_remote_code=True,
            attn_implementation=attn,
            torch_dtype=dtype,
            device_map=device_map,
        )
    else:
        with init_empty_weights():
            model_obj = AutoModelForCausalLM.from_pretrained(
                name_or_path,
                trust_remote_code=True,
                attn_implementation=attn,
                torch_dtype=dtype,
            )
        if mem_limit:
            max_mem = {gpu[0]: f"{mem_limit}GiB"}
            for i in gpu[1:]:
                max_mem[i] = torch.cuda.get_device_properties(i).total_memory
        else:
            max_mem = {i: torch.cuda.get_device_properties(i).total_memory for i in gpu}
        dmap = infer_auto_device_map(
            model_obj,
            max_memory=max_mem,
            no_split_module_classes=no_split,
        )
        model_obj = load_checkpoint_and_dispatch(model_obj, name_or_path, dmap)

    if tokenizer.pad_token is None:
        if any(x in str(name_or_path).lower() for x in ["llama-2", "llama-3", "vicuna"]):
            print("Setting pad token to <pad> for Llama/Vicuna.")
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        else:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        if tokenizer.pad_token == tokenizer.unk_token:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            tokenizer.pad_token = "<pad>"
            model_obj.resize_token_embeddings(len(tokenizer))
        print("tokenizer.pad_token:", tokenizer.pad_token)
        print("tokenizer.pad_token_id:", tokenizer.pad_token_id)
        print("tokenizer.eos_token:", tokenizer.eos_token)

    # Assign chat template using unified table
    set_chat_template(tokenizer, name_or_path)

    if len(tokenizer) != model_obj.config.vocab_size:
        print("Tokenizer vocab differs from model config; resizing embeddings.")
        model_obj.resize_token_embeddings(len(tokenizer))
        model_obj.config.vocab_size = len(tokenizer)

    if tokenizer.pad_token_id is not None:
        model_obj.config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = "left"
    return tokenizer, model_obj


def assert_dialog(dialog):
    """Validate dialog format: alternating user/assistant with optional initial system."""
    assert isinstance(dialog, list)
    for i, msg in enumerate(dialog):
        assert isinstance(msg, dict)
        assert "role" in msg and "content" in msg
        assert msg["role"] in ["user", "assistant", "system"]
        if msg["role"] == "system":
            continue
        # Enforce alternating sequence after system (if any)
        expected_role = "user" if (i == 0 or dialog[i-1]["role"] == "assistant" or dialog[i-1]["role"] == "system") else "assistant"
        assert msg["role"] == expected_role, f"Role mismatch at position {i}: expected {expected_role}, got {msg['role']}"


def get_system_prompt(rate_core: str) -> str:
    """Return system prompt based on evaluation core type."""
    if rate_core == "empathy":
        return "The user is telling you something now, please respond to it."
    elif rate_core in {"truthful", "harmlessness", "helpfulness"}:
        return "The user is asking you a question. Please respond to it."
    elif rate_core == "reasoning":
        return "Please help the user summarize a dialog."
    else:
        raise ValueError(f"Unsupported rate_core type: {rate_core}")


def get_context_pair_dialogs(
    queries: List[str],
    nc_prompts: List[str],
    contexts: List[str],
    put_context_first: bool = False,
    rate_core: str = "empathy"
):
    """
    Build paired dialogs with and without injected context.

    If reasoning core: rewrites prompt to summary instruction.
    """
    def return_pair(query, nc_prompt, context, put_context_first):
        system_prompt = get_system_prompt(rate_core)
        d_nc = [
            {'role': 'system', 'content': system_prompt},
            {"role": "user", "content": nc_prompt}
        ]
        if put_context_first:
            d = [
                {'role': 'system', 'content': system_prompt},
                {"role": "user", "content": context + '\n' + nc_prompt}
            ]
        else:
            if rate_core == "reasoning":
                nc_prompt = f"Dialog:\n{query}\n"
                context = context + "\nNow, please summarize the above dialog in the format: ['Summary': '...', 'Topic': '...']\n"
            d = [
                {'role': 'system', 'content': system_prompt},
                {"role": "user", "content": nc_prompt + '\n' + context}
            ]
        return d, d_nc

    assert len(queries) == len(contexts)
    dialogs, dialogs_nc = [], []
    for query, nc_prompt, context in zip(queries, nc_prompts, contexts):
        cd, ncd = return_pair(query, nc_prompt, context, put_context_first)
        dialogs.append(cd)
        dialogs_nc.append(ncd)
    return dialogs, dialogs_nc


def convert_chat_prompt_to_string(messages, tokenizer):
    """
    Restore the original function's `tokenizer.apply_chat_template` logic,
    ensuring behavior matches the original exactly.
    Note: the original mentioned adding a model_path parameter for tokenizer configuration.
    """
    # Call apply_chat_template to convert to a string
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # only convert to string, do not tokenize
        add_special_tokens=True  # retain special tokens (e.g., bos/eos)
    )


def clean_text(text):
    """
    Clean repeated noisy segments while retaining useful content.
    Truncates to 3000 chars, collapses excessive repetition, normalizes whitespace.
    """
    if not text or not isinstance(text, str):
        return ""
    cleaned = text[:3000]
    for length in range(10, 0, -1):
        pattern = re.compile(rf'(.{{{length}}})\1{{3,}}')
        cleaned = pattern.sub(r'\1\1', cleaned)
    cleaned = re.sub(r'(\b\w+\b)\s+\1{2,}', r'\1', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned





def configure_tokenizer_llm_for_chat(model_path, tensor_parallel_size: int = 1) -> LLM:
    """
    Configure tokenizer and vLLM instance for chat usage (pad + template alignment).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    lower_path = model_path.lower()

    if tokenizer.pad_token is None:
        if any(x in lower_path for x in ["llama-2", "llama-3", "vicuna"]):
            print("Setting pad token to eos for Llama/Vicuna models.")
            tokenizer.pad_token = tokenizer.eos_token
            llm = LLM(
                model=model_path,
                gpu_memory_utilization=0.9,
                max_model_len=4096,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True,
            )
        else:
            print("Unknown model; using eos_token as pad.")
            tokenizer.pad_token = tokenizer.eos_token
            llm = LLM(
                model=model_path,
                max_model_len=4096,
                tokenizer_kwargs={"pad_token": tokenizer.eos_token, "pad_token_id": tokenizer.eos_token_id},
            )
    else:
        if tokenizer.pad_token == tokenizer.unk_token:
            print("Replacing pad_token (unk) with </s>.")
            tokenizer.pad_token = "</s>"
            tokenizer.pad_token_id = 2
            tokenizer.padding_side = "left"
            temp_dir = f"/tmp/custom_tokenizer_{os.path.basename(model_path)}"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            print(f"Modified tokenizer saved to: {temp_dir}")
            llm = LLM(
                model=model_path,
                tokenizer=temp_dir,
                gpu_memory_utilization=0.9,
                max_model_len=4096,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True,
            )
        else:
            llm = LLM(
                model=model_path,
                gpu_memory_utilization=0.9,
                max_model_len=4096,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True,
            )

    # Assign template
    set_chat_template(tokenizer, model_path)
    tokenizer.padding_side = "left"
    return tokenizer, llm
