import os
import re
import argparse
import ollama
from tabulate import tabulate

FEW_SHOT_EXAMPLES = [
    {"text": "Senate Approves New Infrastructure Spending Bill.", "label": "Politics"},
    {"text": "Prime Minister Announces Snap Election Amid Coalition Collapse.", "label": "Politics"},
    {"text": "Governor Signs Executive Order Restricting Industrial Emissions.", "label": "Politics"},
    {"text": "New Quantum Computing Chip Breakthrough Multiplies Processing Speeds.", "label": "Technology"},
    {"text": "Cybersecurity Breach Exposes Millions of User Accounts Worldwide.", "label": "Technology"},
    {"text": "Startup Launches Revolutionary Virtual Reality Headset for Remote Workers.", "label": "Technology"},
    {"text": "Underdog Team Secures Dramatic Victory in Championship Final.", "label": "Sports"},
    {"text": "Olympic Committee Announces New Host City for Upcoming Summer Games.", "label": "Sports"},
    {"text": "World Number One Tennis Star Withdraws From Tournament Due to Injury.", "label": "Sports"},
    {"text": "Stock Market Plummets as Tech Sector Experiences Massive Sell-Off.", "label": "Finance"},
    {"text": "Global Conglomerate Reports Record Profits in Q3 Financial Release.", "label": "Finance"},
    {"text": "Cryptocurrency Regulations Tighten Across European Markets.", "label": "Finance"},
    {"text": "Study Finds Regular Exercise Significantly Reduces Risk of Heart Disease.", "label": "Health"},
    {"text": "Hospitals Face Severe Nurse Shortages Amid Seasonal Flu Surge.", "label": "Health"},
    {"text": "Researchers Map Human Genome Sequence to Uncover Rare Genetic Mutations.", "label": "Health"}
]

VALID_CATEGORIES = ["Politics", "Technology", "Sports", "Finance", "Health"]

SYNTHETIC_DATASET = [
    {"headline": "Tech Giants Agree on New Open-Source AI Safety Standards", "label": "Technology"},
    {"headline": "Central Bank Raises Interest Rates by 25 Basis Points to Combat Inflation", "label": "Finance"},
    {"headline": "Star Striker Signs Record-Breaking Five-Year Contract Extension", "label": "Sports"},
    {"headline": "Parliament Votes to Pass Historic Climate Action Bill After Fierce Debate", "label": "Politics"},
    {"headline": "New FDA-Approved Breakthrough Drug Shows Promise in Halting Alzheimer's", "label": "Health"},
]

def clean_label(raw_output: str) -> str:
    """Helper utility to extract and validate categories from text output."""
    for category in VALID_CATEGORIES:
        if re.search(rf"\b{category}\b", raw_output, re.IGNORECASE):
            return category
    return "Unknown"

def classify_zero_shot(headline: str, model_name: str) -> tuple[str, int, int]:
    system_prompt = (
        "You are an expert news editor. Your task is to classify the provided headline "
        f"into exactly one of these categories: {VALID_CATEGORIES}. "
        "Respond with ONLY the category name. Do not write markdown, intros, or punctuation."
    )
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Headline: {headline}"}
        ],
        options={"temperature": 0.0}
    )
    raw_content = response['message']['content'].strip()
    return clean_label(raw_content), response.get('prompt_eval_count', 0), response.get('eval_count', 0)

def classify_few_shot(headline: str, examples: list[dict], model_name: str) -> tuple[str, int, int]:
    system_prompt = (
        f"Classify the input headline into one of these categories: {VALID_CATEGORIES}. "
        "Follow the exact format shown in the examples. Provide ONLY the category name."
    )
    example_blocks = [f"Headline: {ex['text']}\nCategory: {ex['label']}" for ex in examples]
    user_content = "\n\n".join(example_blocks) + f"\n\nHeadline: {headline}\nCategory:"
    
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        options={"temperature": 0.0}
    )
    raw_content = response['message']['content'].strip()
    return clean_label(raw_content), response.get('prompt_eval_count', 0), response.get('eval_count', 0)

def classify_cot(headline: str, model_name: str) -> tuple[str, int, int]:
    system_prompt = (
        f"Classify the news headline into one of: {VALID_CATEGORIES}. "
        "First, reason step-by-step about what fields or industries the words in the headline relate to. "
        "Finally, end your response with the phrase 'Final Category: <label>'."
    )
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Headline: {headline}"}
        ],
        options={"temperature": 0.2} 
    )
    raw_content = response['message']['content'].strip()
    final_line_match = re.search(r"Final Category:\s*(\w+)", raw_content, re.IGNORECASE)
    label = clean_label(final_line_match.group(1)) if final_line_match else clean_label(raw_content)
    return label, response.get('prompt_eval_count', 0), response.get('eval_count', 0)


def load_data(input_file: str, labels_file: str) -> list[dict]:
    """Loads headlines and ground truth from files, or returns synthetic fallback."""
    if input_file and labels_file:
        try:
            with open(input_file, 'r', encoding='utf-8') as f_in, open(labels_file, 'r', encoding='utf-8') as f_lbl:
                headlines = [line.strip() for line in f_in if line.strip()]
                labels = [line.strip() for line in f_lbl if line.strip()]
            
            if len(headlines) != len(labels):
                print(f"Warning: Mismatch in lines! {len(headlines)} headlines vs {len(labels)} labels.")
                
            return [{"headline": h, "label": l} for h, l in zip(headlines, labels)]
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            exit(1)
    else:
        print("No input files provided via arguments. Using built-in synthetic dataset.")
        return SYNTHETIC_DATASET

def run_evaluation(dataset: list[dict], model_name: str):
    print(f"\nInitializing Prompt Engineering Playground...")
    print(f"Model: '{model_name}' | Total Headlines: {len(dataset)}\n")
    
    metrics = {
        "Zero-Shot": {"correct": 0, "prompt_tok": 0, "comp_tok": 0},
        "Few-Shot": {"correct": 0, "prompt_tok": 0, "comp_tok": 0},
        "Chain-of-Thought": {"correct": 0, "prompt_tok": 0, "comp_tok": 0}
    }
    
    total_items = len(dataset)
    
    for idx, item in enumerate(dataset, 1):
        headline = item["headline"]
        ground_truth = item["label"]
        print(f"Processing [{idx}/{total_items}]: \"{headline[:40]}...\"")
        
        zs_pred, zs_p_tok, zs_c_tok = classify_zero_shot(headline, model_name)
        metrics["Zero-Shot"]["prompt_tok"] += zs_p_tok
        metrics["Zero-Shot"]["comp_tok"] += zs_c_tok
        if zs_pred == ground_truth: metrics["Zero-Shot"]["correct"] += 1
        
        fs_pred, fs_p_tok, fs_c_tok = classify_few_shot(headline, FEW_SHOT_EXAMPLES, model_name)
        metrics["Few-Shot"]["prompt_tok"] += fs_p_tok
        metrics["Few-Shot"]["comp_tok"] += fs_c_tok
        if fs_pred == ground_truth: metrics["Few-Shot"]["correct"] += 1
        
        cot_pred, cot_p_tok, cot_c_tok = classify_cot(headline, model_name)
        metrics["Chain-of-Thought"]["prompt_tok"] += cot_p_tok
        metrics["Chain-of-Thought"]["comp_tok"] += cot_c_tok
        if cot_pred == ground_truth: metrics["Chain-of-Thought"]["correct"] += 1

    print("\n" + "="*70)
    print("                 PROMPT DESIGN EVALUATION RESULTS                 ")
    print("="*70 + "\n")
    
    table_data = []
    for approach, data in metrics.items():
        accuracy = (data["correct"] / total_items) * 100
        avg_prompt = data["prompt_tok"] / total_items
        avg_comp = data["comp_tok"] / total_items
        avg_total = avg_prompt + avg_comp
        
        table_data.append([
            approach, f"{data['correct']}/{total_items}", f"{accuracy:.1f}%",
            f"{avg_prompt:.1f}", f"{avg_comp:.1f}", f"{avg_total:.1f}"
        ])
        
    headers = ["Approach", "Correct", "Accuracy", "Avg Prompt Tok", "Avg Gen Tok", "Avg Total Tok"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def main():
    parser = argparse.ArgumentParser(
        description="News Classifier CLI: Compare Zero-Shot, Few-Shot, and CoT prompting.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="llama3",
        help="Ollama model to use for classification (default: llama3)"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Path to the plain text file containing news headlines (one per line)"
    )
    parser.add_argument(
        "-l", "--labels",
        type=str,
        help="Path to the plain text file containing ground truth labels (one per line)"
    )
    
    args = parser.parse_args()
    
    if bool(args.input) != bool(args.labels):
        parser.error("You must provide both --input and --labels, or neither (to use fallback data).")
    
    dataset = load_data(args.input, args.labels)
    run_evaluation(dataset, args.model)

if __name__ == "__main__":
    main()