import csv
import os
import json
import time
from tqdm import tqdm
import tiktoken
from transformers import pipeline, set_seed
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

generator = pipeline(
    'text-generation', model='openai-community/gpt2', device=device)
set_seed(42)

MAX_TOKENS = 1024  # GPT-2 has a context window of 1024 tokens
INITIAL_BATCH_SIZE = 10
MIN_BATCH_SIZE = 5
MAX_BATCH_SIZE = 20


def num_tokens_from_string(string: str, encoding_name: str = "gpt2") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))


def truncate_text(text, max_length=300):
    if text is None:
        return ""
    return text if len(text) <= max_length else text[:max_length] + "..."


def classify_posts_batch(posts):
    prompt = "Classify each of the following Reddit posts into one of three categories:\n"
    prompt += "1. Is it AI? (posts about detecting AI-generated images)\n"
    prompt += "2. How do I make this? (posts asking how to create specific images with AI)\n"
    prompt += "3. Other (everything else)\n\n"

    for i, post in enumerate(posts, 1):
        prompt += f"Post {i}:\nTitle: {truncate_text(post.get('title', ''))}\nContent: {truncate_text(post.get('selftext', ''))}\n\n"

    prompt += "Respond with only the category numbers (1, 2, or 3) for each post, separated by commas."

    try:
        generated = generator(prompt, max_length=len(
            prompt.split()) + 50, num_return_sequences=1)[0]['generated_text']
        categories = generated[len(prompt):].strip().split(',')
        return [
            {
                "1": "Is it AI?",
                "2": "How do I make this?",
                "3": "Other"
            }.get(cat.strip(), "Other") for cat in categories
        ]
    except Exception as e:
        print(f"Error in classify_posts_batch: {str(e)}")
        return ["Error"] * len(posts)


def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            return {
                'last_processed_row': checkpoint.get('last_processed_row', 0),
                'current_batch_size': checkpoint.get('current_batch_size', INITIAL_BATCH_SIZE)
            }
    return {'last_processed_row': 0, 'current_batch_size': INITIAL_BATCH_SIZE}


def save_checkpoint(checkpoint_file, last_processed_row, current_batch_size):
    with open(checkpoint_file, 'w') as f:
        json.dump({'last_processed_row': last_processed_row,
                  'current_batch_size': current_batch_size}, f)


def process_csv(input_file, output_file, checkpoint_file='checkpoint.json'):
    try:
        checkpoint = load_checkpoint(checkpoint_file)
        last_processed_row = checkpoint['last_processed_row']
        current_batch_size = checkpoint['current_batch_size']

        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            total_rows = sum(1 for row in csv.DictReader(infile))

        with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
                open(output_file, 'a', newline='', encoding='utf-8') as outfile:

            reader = csv.DictReader(infile)
            fieldnames = ['category'] + reader.fieldnames
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)

            if last_processed_row == 0:
                writer.writeheader()

            # Skip already processed rows
            for _ in range(last_processed_row):
                next(reader, None)

            # Create a progress bar
            pbar = tqdm(total=total_rows, initial=last_processed_row,
                        desc="Processing", unit="row")

            try:
                batch = []
                start_time = time.time()
                processed_since_last_checkpoint = 0

                for row in reader:
                    batch.append(row)

                    if len(batch) == current_batch_size:
                        prompt = "Classify each post:\n" + \
                            "\n".join(
                                [f"Title: {truncate_text(post.get('title', ''))}\nContent: {truncate_text(post.get('selftext', ''))}" for post in batch])
                        token_count = num_tokens_from_string(prompt)

                        if token_count > MAX_TOKENS:
                            current_batch_size = max(
                                MIN_BATCH_SIZE, current_batch_size // 2)
                            print(
                                f"Reduced batch size to {current_batch_size}")
                            continue

                        categories = classify_posts_batch(batch)
                        for post, category in zip(batch, categories):
                            new_row = {'category': category}
                            new_row.update(post)
                            writer.writerow(new_row)
                            last_processed_row += 1
                            processed_since_last_checkpoint += 1
                            pbar.update(1)

                        batch = []

                        if token_count < MAX_TOKENS // 2 and current_batch_size < MAX_BATCH_SIZE:
                            current_batch_size = min(
                                MAX_BATCH_SIZE, current_batch_size * 2)
                            print(
                                f"Increased batch size to {current_batch_size}")

                        # Log performance metrics every 1000 processed rows
                        if processed_since_last_checkpoint >= 1000:
                            elapsed_time = time.time() - start_time
                            rows_per_second = processed_since_last_checkpoint / elapsed_time
                            print(f"\nPerformance update:")
                            print(
                                f"Rows processed: {processed_since_last_checkpoint}")
                            print(f"Time elapsed: {elapsed_time:.2f} seconds")
                            print(
                                f"Processing speed: {rows_per_second:.2f} rows/second")
                            print(f"Current batch size: {current_batch_size}")

                            save_checkpoint(
                                checkpoint_file, last_processed_row, current_batch_size)
                            start_time = time.time()
                            processed_since_last_checkpoint = 0

                if batch:
                    categories = classify_posts_batch(batch)
                    for post, category in zip(batch, categories):
                        new_row = {'category': category}
                        new_row.update(post)
                        writer.writerow(new_row)
                        last_processed_row += 1
                        pbar.update(1)

            except KeyboardInterrupt:
                print("\nScript interrupted. Progress saved.")
            finally:
                save_checkpoint(checkpoint_file,
                                last_processed_row, current_batch_size)
                pbar.close()

    except Exception as e:
        print(f"Error in process_csv: {str(e)}")
        raise


if __name__ == "__main__":
    input_file = "updated_file.csv"
    output_file = "AI-labeled-reddit-posts.csv"
    process_csv(input_file, output_file)
    print(f"Processing complete. Results saved to {output_file}")
