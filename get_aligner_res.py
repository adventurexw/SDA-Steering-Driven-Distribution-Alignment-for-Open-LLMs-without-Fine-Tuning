"""
Script to obtain aligner-corrected responses using vLLM for batch inference.
This has dependencies and requires original responses to be available first.
"""

import os
import argparse
import json
import tqdm
from vllm import LLM, SamplingParams
import gc
import torch

# Assume these functions are imported from the original sda.response module;
# if they don't exist, implement them yourself.
from sda.response import save_responses_to_jsonl, make_aligner_prompt, truncate_response_by_marker
from sda.utils import clean_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_model', type=str, required=True, help="Original model name")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the original dataset")
    parser.add_argument('--aligner_model_path', type=str, required=True, help="Path to the aligner model")
    parser.add_argument('--input_jsonl', type=str, required=True, help="Path to jsonl file containing query and original_response")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save jsonl file with aligner corrections")
    parser.add_argument('--rate_core', type=str, required=True, help="Rating core parameter, keep consistent with the original script")
    parser.add_argument('--gpus', type=str, default="0", help="GPU IDs to use, e.g. '0,1'")
    parser.add_argument('--max_tokens', type=int, default=4096, help="Maximum tokens for generated responses")
    parser.add_argument('--T0', type=float, default=0.7, help="Sampling temperature")
    parser.add_argument('--top_p', type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for inference")
    args = parser.parse_args()

    print("********  Get aligner responses   ********")
    # Print argument information
    print(f"Aligner model path: {args.aligner_model_path}")
    print(f"Input file: {args.input_jsonl}")
    print(f"Output file: {args.output_path}")
    print(f"Using GPUs: {args.gpus}")
    print(f"Batch size: {args.batch_size}")

    # Print configuration info
    print("original_response_model:", args.original_model)
    dataset_name = os.path.basename(args.dataset_path)
    print("Dataset path:", args.dataset_path)
    print("Dataset:", dataset_name)
    print("Rate core:", args.rate_core)
    print("Batch size:", args.batch_size)
    print("Temperature (T0):", args.T0)
    print("Top_p:", args.top_p)

    # Create output directory
    a_responses_dir = os.path.dirname(args.output_path)
    os.makedirs(a_responses_dir, exist_ok=True)
    jsonl_res_name = f"AlignerCorrect_{args.original_model}_{dataset_name[:-6]}_{args.rate_core}_aligner_responses.jsonl"
    a_responses_file = os.path.join(a_responses_dir, jsonl_res_name)

    # Load input data
    print("Loading input data...")
    data = []
    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            # Validate required fields
            if "query" not in item or "original_response" not in item:
                print(f"Skipping invalid item: {line.strip()[:50]}...")
                continue
            data.append(item)
    print(f"Loaded {len(data)} valid items")

    # Set visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print(f"Using GPUs: {args.gpus}")

    # Prepare vLLM model
    sampling_params = SamplingParams(
        temperature=args.T0,
        top_p=args.top_p,
        max_tokens=4096
    )

    print("Loading aligner model with vLLM...")
    llm = LLM(
        model=args.aligner_model_path,
        tensor_parallel_size=len(args.gpus.split(',')),
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )

    # Prepare batched data
    prompts = []  # store final strings to feed into vLLM
    items = []    # store original items for later saving
    count = 0

    print("Generating aligner corrected responses...")
    for item in tqdm.tqdm(data, desc="Processing queries"):
        count += 1
        # Create a prompt without instruction
        query = item["query"]
        # Build aligner input prompt
        original_response = item["original_response"]
        original_response = original_response[:4096]
        # When the original response exceeds ~5000 chars, vLLM may error; limit to 5000 chars (we use 4096 here)
        aligner_prompt = make_aligner_prompt(query, original_response, args.rate_core)
        prompts.append(aligner_prompt)
        items.append(item)

        # When batch is full or finished, run inference
        if (len(prompts) == args.batch_size) or (count == len(data) and len(prompts) != 0):
            # Batch generate
            outputs = llm.generate(prompts, sampling_params)

            # Process generated results
            data_to_save = []
            for output, item in zip(outputs, items):
                correction = output.outputs[0].text.strip()
                # Save result
                finished_reason = output.outputs[0].finish_reason
                # Note: vLLM's output attribute is finish_reason, not finished_reason
                if finished_reason == "length":
                    print(f"Warning: Response for query '{item['query']}' was truncated due to length limit.")
                    correction = correction[:4096]
                correction = clean_text(correction)  # Clean duplicate content and garbled text in the reply
                item["correction_aligner"] = correction
                data_to_save.append(item)
            save_responses_to_jsonl(data_to_save, a_responses_file)

            # Clear batch data
            prompts = []
            items = []

            # Clear GPU cache
            torch.cuda.empty_cache()
            gc.collect()

    print(f"All processing complete. Results saved to {a_responses_file}")

if __name__ == "__main__":
    main()