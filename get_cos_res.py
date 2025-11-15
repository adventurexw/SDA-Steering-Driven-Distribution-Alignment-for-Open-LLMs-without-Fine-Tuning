"""
sdaJS can only be implemented using the transformers library. vllm can only obtain the top20 logits,
so it cannot compute the full logits distribution and therefore cannot compute s'da.
"""
import os
import argparse
import json
import tqdm
from sda.response import load_dataset, write_nc_prompt, save_responses_to_jsonl, write_instruction, batch_generate_instruction_response, load_base_model
import warnings
warnings.filterwarnings("ignore", message="A decoder-only architecture is being used, but right-padding was detected!")
from sda.utils import clean_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to base model")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--responses_path', type=str, required=True, help="Path to save responses jsonl after run")
    parser.add_argument('--start_line', type=int, default=0, help="Start line")
    parser.add_argument('--end_line', type=int, default=None, help="End line")
    parser.add_argument('--instruction_core', type=str, required=True, help="instruction core")
    parser.add_argument('--rate_core', type=str, required=True, help="Rating core")
    parser.add_argument('--gpus', type=str, default="0,1", help="Comma-separated list of GPU IDs to use")
    parser.add_argument('--T0', type=float, default=0.6, help="Sampling temperature (corresponds to original T0)")
    parser.add_argument('--lambda_function', type=int, default=0, help="Lambda mapping function choice, 0 or 1")
    parser.add_argument('--T_function', type=int, default=0, help="Temperature mapping function choice, 0 or 1")
    parser.add_argument('--is_chat', action='store_true', help="Whether to use chat mode (corresponds to original is_chat)")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size, default 1 to avoid OOM")
    parser.add_argument('--lambda_file', type=str, required=True, help="Path to jsonl file containing lambda information")
    args = parser.parse_args()

    print("*********Starting sdaJS response generation...*********")
    # Print configuration info
    print("Base_model_path:", args.model_path)
    model_name = os.path.basename(args.model_path)
    print("Model name:", model_name)
    dataset_name = os.path.basename(args.dataset_path)
    print("Dataset path:", args.dataset_path)
    print("Responses path:", args.responses_path)
    print("Dataset:", dataset_name)
    print("instruction core:", args.instruction_core)
    print("Rate core:", args.rate_core)
    print("data range:", args.start_line, "-", args.end_line)
    print("Temperature (T0):", args.T0)
    print("Chat mode:", args.is_chat)
    print("Batch size:", args.batch_size)

    # Set multi-GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print(f"Using GPUs: {args.gpus}")

    # Create output directory and file (create in advance to avoid permission issues)
    responses_dir = os.path.dirname(args.responses_path)
    os.makedirs(responses_dir, exist_ok=True)
    jsonl_res_name = f"sdaJS_Correction_{model_name}_{dataset_name[:-6]}_{args.instruction_core}_T0({args.T0})_hyper_k=2_LF({args.lambda_function})_TF({args.T_function})_sdaJS_responses_{args.start_line}-{args.end_line}.jsonl"
    responses_file = os.path.join(responses_dir, jsonl_res_name)
    # Initialize file (clear or create)
    with open(responses_file, 'w', encoding='utf-8') as f:
        pass  # Only create/clear file

    # Load dataset (by specified line range)
    data = load_dataset(args.dataset_path, args.start_line, args.end_line)
    print(f"Read {len(data)} entries from dataset (line range: {args.start_line}-{args.end_line})")

    # Determine question title field
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

    # Load model and tokenizer
    print("Loading tokenizer ")
    print("Loading model with huggingface transformers...")
    model, tokenizer = load_base_model(args.model_path)

    # Read lambda and score info from jsonl file (by specified lines)
    lambda_data = []
    with open(args.lambda_file, 'r') as f:
        for line_num, line in enumerate(f, start=0):
            if (line_num >= args.start_line) and (args.end_line is None or line_num < args.end_line):
                try:
                    lambda_info = json.loads(line.strip())
                    lambda_data.append(lambda_info)
                except json.JSONDecodeError:
                    print(f"Warning: lambda_file line {line_num} is malformed, skipping")
                    continue

    # Verify matching data counts
    if len(lambda_data) != len(data):
        raise ValueError(f"Number of lambda entries ({len(lambda_data)}) does not match dataset entries ({len(data)}). Please check the line range")

    # Prepare all data lists (before batching)
    all_queries = []
    all_prompts = []
    all_instruction_prompts = []
    all_lambdas = []
    all_scores = []
    all_lambda_info = []  # keep original lambda info for saving results

    for lambda_info in lambda_data:
        query = lambda_info["query"]
        nc_prompt = write_nc_prompt(query, args.instruction_core)
        instruction_prompt = write_instruction(args.instruction_core)
        
        all_queries.append(query)
        all_prompts.append(nc_prompt)
        all_instruction_prompts.append(instruction_prompt)
        if "lambda" in lambda_info.keys():
            all_lambdas.append(lambda_info["lambda"])
        else:
            all_lambdas.append(lambda_info["Lambda"])
        all_scores.append(lambda_info["score"])
        all_lambda_info.append(lambda_info)

    # Compute total batches
    total_data = len(all_queries)
    total_batches = (total_data + args.batch_size - 1) // args.batch_size  # ceil division
    print(f"Total entries: {total_data}, Batch size: {args.batch_size}, Total batches: {total_batches}")

    # Process in batches
    for batch_idx in tqdm.tqdm(range(total_batches), desc="Processing batches"):
        # Compute start and end indices for current batch
        start = batch_idx * args.batch_size
        end = min((batch_idx + 1) * args.batch_size, total_data)
        
        # Extract current batch data
        batch_queries = all_queries[start:end]
        batch_prompts = all_prompts[start:end]
        batch_instructions = all_instruction_prompts[start:end]
        batch_lambdas = all_lambdas[start:end]
        batch_scores = all_scores[start:end]
        batch_lambda_info = all_lambda_info[start:end]

        # Call batch generation function to handle current batch
        batch_corrections = batch_generate_instruction_response(
            model, tokenizer, batch_queries, batch_prompts, batch_instructions, batch_lambdas, 
            args.rate_core, args.batch_size,  # pass batch size (same as current batch)
            max_gen_len=4096, T0=args.T0, lambda_function=args.lambda_function,
            T_function=args.T_function, is_chat=args.is_chat
        )

        # Verify number of generated results
        if len(batch_corrections) != len(batch_queries):
            print(f"Warning: Batch {batch_idx} generated count mismatch (expected: {len(batch_queries)}, actual: {len(batch_corrections)})")

        # Save current batch results (optimized: write entire batch at once)
        batch_data_to_save = []  # accumulate all data for current batch
        for i in range(len(batch_queries)):
            # Construct single record
            single_data = {
                "query": batch_queries[i],
                "original_response": batch_lambda_info[i].get("original_response", ""),
                "rate_core": args.rate_core,
                "score": batch_scores[i],
                "Lambda": batch_lambdas[i],
                "correction_sdaJS": clean_text(batch_corrections[i]) if i < len(batch_corrections) else ""
            }
            batch_data_to_save.append(single_data)

        # Write entire batch at once (only one file operation)
        save_responses_to_jsonl(batch_data_to_save, responses_file)

        print(f"Batch {batch_idx + 1}/{total_batches} processed, saved {end - start} entries")

    print(f"All batches processed. Results saved to: {responses_file}")


if __name__ == "__main__":
    main()