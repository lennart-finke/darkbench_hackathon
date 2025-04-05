import json
import os
import argparse
from typing import Dict, List, Any

def parse_log_file(filepath: str) -> Dict[str, Any]:
    """Parse the log file and return the JSON data."""
    with open(filepath, "r") as f:
        return json.loads(f.read())

def process_samples(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process samples from the log data and format them as message pairs.

    Handles multiple JSON lines in content, including code blocks.
    """
    results = []
    for sample in data["samples"]:
        output = sample["output"]["choices"][0]["message"]["content"]
        pattern = sample["target"]

        # Strip code block markers and empty lines
        lines = [
            line.strip() for line in output.split('\n')
            if line.strip() and line.strip() not in ["```", "```json", "```jsonl"]
        ]

        found_valid_json = False
        # Process each line as a potential JSON object
        for line in lines:
            try:
                obj = json.loads(line)
                question = obj.get('question', '')
                answer = obj.get('answer', '')

                if question and answer:
                    results.append({
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer}
                        ],
                        "metadata": {"pattern": pattern}
                    })
                    found_valid_json = True
            except json.JSONDecodeError:
                print(f"  - Failed to parse output line as JSON: {line}")
                continue

        if not found_valid_json:
            print(f"Failed to parse any JSON from: {output}")

    return results

def save_results(results: List[Dict[str, Any]], output_file: str = "data/darkbench_examples.jsonl") -> None:
    """Save the results to a JSONL file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')

    print(f"Processed {len(results)} examples")
    print(f"Results saved to {output_file}")

def main() -> None:
    """Main function to parse the log file and save the results."""
    parser = argparse.ArgumentParser(description="Parse examples from a json log file")
    parser.add_argument("filepath", help="Path to the json log file to parse")
    parser.add_argument("--output", "-o", default="data/darkerbench.jsonl",
                       help="Output JSONL file path (default: data/darkerbench.jsonl)")
    args = parser.parse_args()

    data = parse_log_file(args.filepath)
    results = process_samples(data)
    save_results(results, args.output)

if __name__ == "__main__":
    main()
