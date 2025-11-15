"""
Script to obtain original responses using vLLM for batch inference
"""

import os
import argparse
import json
import tqdm
from functools import partial
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer  # used to reuse original tokenizer's chat template logic
from sda.response import load_dataset, write_nc_prompt, save_responses_to_jsonl, truncate_response_by_marker
from sda.utils import get_system_prompt, configure_tokenizer_llm_for_chat, convert_chat_prompt_to_string, clean_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Base model path")
    parser.add_argument('--dataset_path', type=str, required=True, help="Dataset path")
    parser.add_argument('--responses_path', type=str, required=True, help="Path to save original responses jsonl file")
    parser.add_argument('--start_line', type=int, default=0, help="Start line")
    parser.add_argument('--end_line', type=int, default=None, help="End line")
    parser.add_argument('--instruction_core', type=str, required=True, help="instruction core")
    parser.add_argument('--gpus', type=str, default="0,1", help="Comma-separated list of GPU IDs to use")
    parser.add_argument('--T0', type=float, default=0.6, help="Sampling temperature (corresponds to T0)")
    parser.add_argument('--top_p', type=float, default=0.95, help="Sampling parameter (corresponds to top_p)")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--is_chat', action='store_true', help="Use chat mode (corresponds to is_chat)")
    args = parser.parse_args()

    # Print configuration
    print("Base_model_path:", args.model_path)
    model_name = os.path.basename(args.model_path)
    print("Model name:", model_name)
    dataset_name = os.path.basename(args.dataset_path)
    print("Dataset path:", args.dataset_path)
    print("Responses path:", args.responses_path)
    print("Dataset:", dataset_name)
    print("instruction core:", args.instruction_core)
    print("data:", args.start_line, "-", args.end_line)
    print("Batch size:", args.batch_size)
    print("Temperature (T0):", args.T0)
    print("Top_p:", args.top_p)
    print("Chat mode:", args.is_chat)
    
    # Set multi-GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print(f"Using GPUs: {args.gpus}")
    # Compute tensor_parallel_size (equal to number of specified GPUs)
    tensor_parallel_size = len(args.gpus.split(','))
    
    # Create output directory
    responses_dir = os.path.dirname(args.responses_path)
    os.makedirs(responses_dir, exist_ok=True)
    jsonl_res_name = f"{model_name}_{dataset_name[:-6]}_original_responses_{args.start_line}-{args.end_line}.jsonl"
    responses_file = os.path.join(responses_dir, jsonl_res_name)

    # Load dataset
    data = load_dataset(args.dataset_path, args.start_line, args.end_line)
    
    # Determine the question title field
    if "beaver_tails" in dataset_name:
        question_title = "prompt"
    elif "TruthfulQA" in dataset_name:
        question_title = "Question"
    elif "dialogsum" in dataset_name:
        question_title = "dialogue"
    elif "harmfulqa" in dataset_name:
        question_title = "question"
    else:
        question_title = "prompt"
    
    # Load original tokenizer (to restore chat template logic and ensure consistency with original function input) and llm
    print("Loading tokenizer to align chat template...")
    print("Loading model with vLLM...")
    tokenizer, llm = configure_tokenizer_llm_for_chat(args.model_path, tensor_parallel_size=tensor_parallel_size)


    # Prepare batch data
    prompts = []  # store final strings to send to vLLM
    items = []    # store original data items for later saving
    
    print("Generating original responses...")
    count = 0
    for item in tqdm.tqdm(data, desc="Processing queries"):
        count += 1
        query = item[question_title]
        nc_prompt = write_nc_prompt(query, args.instruction_core)  # restore original function's nc_prompt generation logic
        
        # Strictly restore the original function's in_prompt construction logic
        if args.is_chat:
            print("Using chat mode for prompt formatting...")
            system_prompt = get_system_prompt(args.instruction_core)
            in_prompt = [
                {'role': 'system', 'content': system_prompt},
                {"role": "user", "content": nc_prompt}
            ]
            # Convert using original tokenizer's chat template conversion to ensure exact match with original input string
            formatted_prompt = convert_chat_prompt_to_string(in_prompt, tokenizer)
        else:
            # if is_chat=False: use nc_prompt directly as input string
            formatted_prompt = nc_prompt
        
        prompts.append(formatted_prompt)
        items.append(item)
        
        # Run inference when batch size is reached or all data processed
        if (len(prompts) == args.batch_size) or (count == len(data) and len(prompts) != 0):
            # Sampling parameters strictly aligned with original function (T0 and top_p)
            sampling_params = SamplingParams(
                temperature=args.T0,
                top_p=args.top_p,
                max_tokens=4096
            )
            
            # Use vLLM to generate replies in batch
            outputs = llm.generate(prompts, sampling_params)
            
            # Ensure output order matches input (vLLM returns in input order by default, confirm here)
            data_to_save = []
            for output, item in zip(outputs, items):
                # Get generated reply
                original_response = output.outputs[0].text
                finished_reason = output.outputs[0].finish_reason
                if finished_reason == "length":
                    print(f"Warning: Response for query '{item[question_title]}' was truncated due to length limit.")
                    original_response = original_response[:3000]  # truncate to 3000 characters
                original_response = truncate_response_by_marker(original_response, markers=["</think>"]).strip()
                original_response = clean_text(original_response)  # clean duplicate content and garbled characters
                # Save results (fields consistent with original function output)
                data_to_save.append({
                    "query": item[question_title],
                    "original_response": original_response.strip(),
                })
            save_responses_to_jsonl(data_to_save, responses_file)
            
            # Clear batch data
            prompts = []
            items = []

    print(f"Original responses generated and saved to: {responses_file}")

if __name__ == "__main__":
    main()