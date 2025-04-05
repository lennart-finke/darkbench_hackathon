import json
import pandas as pd
import os
import argparse
from typing import Dict, List, Any

def parse_log_file(filepath: str) -> Dict[str, Any]:
    """Parse the log file and return the JSON data."""
    with open(filepath, "r") as f:
        return json.loads(f.read())

def process_samples(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Process samples from the log data and extract relevant information."""
    results = []
    for sample in data["samples"]:
        output = sample["output"]["choices"][0]["message"]["content"]
        pattern = sample["target"]
        output = output.replace("```json", "").replace("```", "").strip()
        try:
            sample_dict = json.loads(output)
            # Extract reasoning and negative_example
            results.append({
                'question': sample_dict.get('question', ''),
                'answer': sample_dict.get('answer', ''),
                'pattern': pattern
            })
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {output}")
            continue
    return results

def save_results(results: List[Dict[str, str]], output_file: str = "data/negative_examples.csv") -> None:
    """Save the results to a CSV file."""
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def main() -> None:
    """Main function to parse the log file and save the results."""
    parser = argparse.ArgumentParser(description="Parse examples from a json log file")
    parser.add_argument("filepath", help="Path to the json log file to parse")
    parser.add_argument("--output", "-o", default="data/negative_examples.csv",
                       help="Output CSV file path (default: data/negative_examples.csv)")
    args = parser.parse_args()

    data = parse_log_file(args.filepath)
    results = process_samples(data)
    save_results(results, args.output)

if __name__ == "__main__":
    main()
