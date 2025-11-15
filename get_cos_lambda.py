"""
Script to obtain Lambda values separately
Lambda corresponds to the amplifying factor 'a' in the paper

Input: jsonl file of original_response (contains query and original_response)
Output: jsonl file containing query, original_response, score, and lambda
"""

import os
import re
import argparse
import json
import tqdm
from sda.response import generate_Lambda, save_responses_to_jsonl
from sda.utils import get_system_prompt,configure_tokenizer_llm_for_chat

def load_original_responses(file_path, start_line=0, end_line=None):
    """Load original responses file (original_response jsonl)"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            # Filter data according to start_line and end_line
            if line_num < start_line:
                continue
            if end_line is not None and line_num >= end_line:
                break
            try:
                item = json.loads(line.strip())
                # Validate required fields
                if "query" in item and "original_response" in item:
                    data.append(item)
                else:
                    print(f"Skipping incomplete data (line {line_num}): missing query or original_response")
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON (line {line_num})")
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_model', type=str, required=True, help="Name of the original model")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the original dataset")
    parser.add_argument('--original_responses_path', type=str, required=True, 
                        help="Path to the original_response jsonl file (input)")
    parser.add_argument('--lambda_output_path', type=str, required=True, 
                        help="Path to save the lambda results jsonl file (output)")
    parser.add_argument('--start_line', type=int, default=0, 
                        help="Start line number to process")
    parser.add_argument('--end_line', type=int, default=None, 
                        help="End line number to process (exclusive)")
    parser.add_argument('--rate_core', type=str, required=True, 
                        help="Rating core parameter (corresponds to generate_Lambda)")
    parser.add_argument('--lambda_function', type=int, default=0, 
                        help="Lambda mapping function selection (0 or 1, corresponds to generate_Lambda)")
    args = parser.parse_args()

    # Print configuration info
    print("Original responses file path:", args.original_responses_path)
    print("Lambda results output path:", args.lambda_output_path)
    print("Processing range:", args.start_line, "-", args.end_line if args.end_line else "all")
    print("rate_core:", args.rate_core)
    print("lambda_function:", args.lambda_function)
    print("original_response_model:", args.original_model)
    dataset_name = os.path.basename(args.dataset_path)
    print("Dataset path:", args.dataset_path)
    print("Dataset:", dataset_name)

    # Create output directory
    responses_dir = os.path.dirname(args.lambda_output_path)
    os.makedirs(responses_dir, exist_ok=True)
    jsonl_lam_name = f"{args.original_model}_{dataset_name[:-6]}_score_lambda_{args.start_line}-{args.end_line}.jsonl"
    score_lam_jsonl = os.path.join(responses_dir, jsonl_lam_name)

    # Load original response data
    print("Loading original responses...")
    original_data = load_original_responses(
        args.original_responses_path,
        args.start_line,
        args.end_line
    )
    print(f"Loaded {len(original_data)} valid records")

    # Record failed items
    fail_data = []
    data_to_save = []

    # Batch compute Lambda
    print("Starting Lambda computation...")
    for item in tqdm.tqdm(original_data, desc="Processing Lambda"):
        try:
            query = item["query"]
            original_response = item["original_response"]
            original_response = re.sub(r'\n+', '\n', original_response.strip())

            # Call generate_Lambda to compute score and lambda
            Lambda, score = generate_Lambda(
                query=query,
                response=original_response.strip('\n'),
                rate_core=args.rate_core,
                lambda_function=args.lambda_function
            )

            if Lambda == -1:
                # Mark as failed
                print(f"Lambda computation failed (query: {query[:50]}...)")
                fail_data.append(item)

            # Construct output item
            output_item = {
                "query": query,
                "original_response": original_response,
                "score": score,
                "lambda": Lambda  # key name lower-case 'lambda' as required
            }
            data_to_save.append(output_item)

        except Exception as e:
            print(f"Error processing item (query: {item.get('query', 'unknown')[:50]}...): {str(e)}")
            fail_data.append(item)
            # Construct output item for failure
            output_item = {
                "query": query,
                "original_response": original_response,
                "score": -1, # mark as failed
                "lambda": -1  # key name lower-case 'lambda' as required
            }
            data_to_save.append(output_item)
        # Results are saved after loop

    save_responses_to_jsonl(data_to_save, score_lam_jsonl)
    # Print processing summary
    total = len(original_data)
    success = total - len(fail_data)
    print(f"Processing complete: {success} succeeded, {len(fail_data)} failed")
    print(f"Lambda results saved to: {args.lambda_output_path}")

    # Save failed data (if any)
    if fail_data:
        fail_path =  os.path.join(args.lambda_output_path ,"_lambda_fail.jsonl")
        with open(fail_path, 'w', encoding='utf-8') as f:
            for item in fail_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Failed items saved to: {fail_path}")

if __name__ == "__main__":
    main()
