import json
import argparse
import os
import glob
import matplotlib.pyplot as plt

from datasets import load_from_disk, load_dataset


STORAGE_PATH = os.getenv("STORAGE_PATH")
SOURCE_DATASET = os.getenv("SOURCE_DATASET")


def load_result_files(result_file):

    all_data = []
    with open(result_file, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))

    visualize_scores(all_data)

    return all_data


def visualize_scores(all_data):
    scores = [item['score'] for item in all_data]
    plt.hist(scores, bins=11)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.savefig(f'{STORAGE_PATH}/generated_question/{args.save_name}_scores.png')


def load_source_data(data_path):
    if os.path.isdir(data_path):
        dataset = load_from_disk(data_path)
    else:
        dataset = load_dataset(data_path, split="train")
    return dataset


def main(args):
    
    args.result_file = f'{STORAGE_PATH}/generated_question/{args.save_name}_results.json'
    args.output_dir = f"{STORAGE_PATH}/local_train_dataset/{args.save_name}"
    args.summary_path = os.path.join(args.output_dir, 'summary.json')

    if os.path.exists(args.output_dir) and os.path.exists(args.summary_path):
        print(f"Local train dataset {args.save_name} already exists, skipping...")
        return

    print(f"[Filter] Loading result files for {args.save_name}...")
    all_data = load_result_files(args.result_file)
    dataset = load_source_data(SOURCE_DATASET)

    # record the mapping between the generated data and the source data
    meta_mapping = [] 

    for item in all_data:
        score = item['score']
        answer = item['answer']
        image_idx = int(item['image_idx'])
        
        if args.min_score <= score <= args.max_score and answer not in ['', 'None', None]:
            meta_mapping.append({
                'dataset_idx': image_idx,
                'question': item['question'],
                'answer': answer,
                'score': score,
                'problem_type': item.get('question_type', 'unknown')
            })

    if not meta_mapping:
        print(f"[Filter] No valid data to process")
        return

    # select the indices of the source data
    indices_to_select = [int(m['dataset_idx']) for m in meta_mapping]
    print(f"[Filter] Selected {len(indices_to_select)} source data")

    # select the source data
    filtered_data = dataset.select(indices_to_select)

    # update the metadata of the filtered data
    def update_row(example, idx):
        info = meta_mapping[idx]
        # keep the 'images' column, only update the text fields
        example['problem'] = info['question']
        example['answer'] = info['answer']
        example['score'] = info['score']
        example['problem_type'] = info['problem_type']
        return example

    # use batched=False with with_indices=True to ensure one-to-one correspondence
    # if the data is large, you can enable num_proc=8
    filtered_data = filtered_data.map(update_row, with_indices=True)

    # remove the columns that are not needed
    cols_to_remove = [col for col in filtered_data.column_names if col not in ['problem', 'answer', 'score', 'images', 'problem_type']]
    filtered_data = filtered_data.remove_columns(cols_to_remove)

    print(f"[Filter] Successfully processed {len(filtered_data)} samples.")

    # Create output directory if specified
    os.makedirs(args.output_dir, exist_ok=True)
    filtered_data.save_to_disk(args.output_dir)
    print(f"[Filter] Saved {len(filtered_data)} samples to {args.output_dir}")

    # Also save a summary file
    summary = {
        "total_images": len(set(indices_to_select)),
        "total_samples": len(filtered_data),
        "score_range": [args.min_score, args.max_score],
        "experiment_name": args.save_name,
    }
    with open(args.summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"[Filter] Saved summary to {args.summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_score", type=float, default=0.7)
    parser.add_argument("--min_score", type=float, default=0.3)
    parser.add_argument("--save_name", type=str, required=True, help="Base name for input and output files")
    args = parser.parse_args()

    main(args)
