import csv
import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import tiktoken

# Load environment variables
load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_TOKENS = 15000  # Set a safe limit below the 16385 token maximum
INITIAL_BATCH_SIZE = 50
MIN_BATCH_SIZE = 10
MAX_BATCH_SIZE = 100

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    # Ignore the '<|endoftext|>' token
    string = string.replace('<|endoftext|>', '')
    try:
        num_tokens = len(encoding.encode(string, disallowed_special=()))
    except Exception as e:
        print(f"Error in tokenizing string: {e}")
        print(f"Problematic string: {string}")
        return 0  # Return 0 tokens for problematic strings
    return num_tokens

def truncate_text(text, max_length=300):
    if text is None:
        return ""
    return text if len(text) <= max_length else text[:max_length] + "..."

def classify_posts_batch(posts):
    prompt = "Classify each of the following Reddit posts into one of three categories:\n"
    prompt += "1. Is it AI? (posts of users trying to detect if an image is AI-generated or not)\n"
    prompt += "2. How do I make this? (posts of users asking what is the prompt to create specific images with AI)\n"
    prompt += "3. Other (everything else)\n\n"
    
    for i, post in enumerate(posts, 1):
        prompt += f"Post {i}:\nTitle: {truncate_text(post.get('title', ''))}\nContent: {truncate_text(post.get('selftext', ''))}\n\n"
    
    prompt += "Respond with only the category numbers (1, 2, or 3) for each post, separated by commas."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that classifies Reddit posts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            n=1,
            temperature=0.5,
        )

        categories = response.choices[0].message.content.strip().split(',')
        return [
            {
                "1": "It's giving AI",
                "2": "Prompt Panhandling",
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
        json.dump({'last_processed_row': last_processed_row, 'current_batch_size': current_batch_size}, f)

def process_csv(input_file, output_file, checkpoint_file='checkpoint.json'):
    try:
        checkpoint = load_checkpoint(checkpoint_file)
        last_processed_row = checkpoint['last_processed_row']
        current_batch_size = checkpoint['current_batch_size']

        # Count the total number of rows
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            total_rows = sum(1 for row in csv.DictReader(infile))

        # Process the file with a progress bar
        with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
             open(output_file, 'a', newline='', encoding='utf-8') as outfile:
            
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames + ['category']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            
            if last_processed_row == 0:
                writer.writeheader()

            # Skip already processed rows
            for _ in range(last_processed_row):
                next(reader, None)

            # Create a progress bar
            pbar = tqdm(total=total_rows, initial=last_processed_row, desc="Processing", unit="row")

            try:
                batch = []
                start_time = time.time()
                processed_since_last_checkpoint = 0
                
                for row in reader:
                    batch.append(row)
                    
                    if len(batch) == current_batch_size:
                        prompt = "Classify each post:\n" + "\n".join([f"Title: {truncate_text(post.get('title', ''))}\nContent: {truncate_text(post.get('selftext', ''))}" for post in batch])
                        token_count = num_tokens_from_string(prompt)
                        
                        if token_count > MAX_TOKENS:
                            current_batch_size = max(MIN_BATCH_SIZE, current_batch_size // 2)
                            print(f"Reduced batch size to {current_batch_size}")
                            continue
                        
                        categories = classify_posts_batch(batch)
                        for post, category in zip(batch, categories):
                            post['category'] = category
                            writer.writerow(post)
                            last_processed_row += 1
                            processed_since_last_checkpoint += 1
                            pbar.update(1)
                        
                        batch = []

                        if token_count < MAX_TOKENS // 2 and current_batch_size < MAX_BATCH_SIZE:
                            current_batch_size = min(MAX_BATCH_SIZE, current_batch_size * 2)
                            print(f"Increased batch size to {current_batch_size}")
                        
                        # Log performance metrics every 1000 processed rows
                        if processed_since_last_checkpoint >= 1000:
                            elapsed_time = time.time() - start_time
                            rows_per_second = processed_since_last_checkpoint / elapsed_time
                            print(f"\nPerformance update:")
                            print(f"Rows processed: {processed_since_last_checkpoint}")
                            print(f"Time elapsed: {elapsed_time:.2f} seconds")
                            print(f"Processing speed: {rows_per_second:.2f} rows/second")
                            print(f"Current batch size: {current_batch_size}")
                            
                            save_checkpoint(checkpoint_file, last_processed_row, current_batch_size)
                            start_time = time.time()
                            processed_since_last_checkpoint = 0

                # Process any remaining posts
                if batch:
                    categories = classify_posts_batch(batch)
                    for post, category in zip(batch, categories):
                        post['category'] = category
                        writer.writerow(post)
                        last_processed_row += 1
                        pbar.update(1)

            except KeyboardInterrupt:
                print("\nScript interrupted. Progress saved.")
            finally:
                save_checkpoint(checkpoint_file, last_processed_row, current_batch_size)
                pbar.close()  # Close the progress bar

    except Exception as e:
        print(f"Error in process_csv: {str(e)}")
        raise  # Re-raise the exception to see the full traceback

if __name__ == "__main__":
    input_file = "sample-dataset-1000-reddit-posts.csv"  # Replace with your input file name
    output_file = "AI-labeled-reddit-posts.csv"
    process_csv(input_file, output_file)
    print(f"Processing complete. Results saved to {output_file}")
    