import json
import pandas as pd
import os

# How to run: Just adjust this filepath.
FILEPATH = "logs/2025-04-05T16-32-20+02-00_evaluate-story-dataset_cyRfyQYnVpJyFuD7qu8ixW.json"

with open(FILEPATH, "r") as f:
    d = json.loads(f.read())

results = []
for sample in d["samples"]:
    output = sample["output"]["choices"][0]["message"]["content"]
    pattern = sample["target"]
    output = output.replace("```json", "").replace("```", "").strip()
    try:
        sample_dict = json.loads(output)
        # Extract reasoning and negative_example
        results.append({
            'reasoning': sample_dict.get('reasoning', ''),
            'negative_example': sample_dict.get('negative_example', ''),
            'pattern': pattern
        })
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {output}")
        continue

# Create dataframe from results
df = pd.DataFrame(results)
os.makedirs("data", exist_ok=True)
df.to_csv("data/negative_examples.csv", index=False)