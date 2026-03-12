import os
import json
import argparse
import pandas as pd


LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
    "0": "entailment",
    "1": "neutral",
    "2": "contradiction",
}


def normalize_label(x):
    if x in LABEL_MAP:
        return LABEL_MAP[x]

    if isinstance(x, str):
        x = x.strip().lower()
        if "entailment" in x or "entails" in x or "implied" in x or "implies" in x:
            return "entailment"
        if "neutral" in x:
            return "neutral"
        if "contradiction" in x or "contradict" in x:
            return "contradiction"

    return None


def compute_accuracy_for_file(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    total = 0
    correct = 0
    invalid = 0

    for _, item in data.items():
        gold = normalize_label(item["label"])
        pred = normalize_label(item["full_response"])

        if pred is None:
            invalid += 1

        if gold is not None and pred is not None and gold == pred:
            correct += 1

        total += 1

    acc = correct / total if total > 0 else 0.0
    invalid_rate = invalid / total if total > 0 else 0.0

    return {
        "file": os.path.basename(filepath),
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "invalid_rate": invalid_rate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing inference json files",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="evaluation_results.csv",
        help="Where to save the summary CSV",
    )
    args = parser.parse_args()

    files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".json")
    ]
    files = sorted(files)

    if not files:
        raise FileNotFoundError(f"No json files found in {args.input_dir}")

    results = []
    for fp in files:
        res = compute_accuracy_for_file(fp)
        results.append(res)

    df = pd.DataFrame(results)
    df["accuracy"] = df["accuracy"].round(4)
    df["invalid_rate"] = df["invalid_rate"].round(4)

    print("\nEvaluation results:")
    print(df.to_string(index=False))

    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved summary to: {args.output_csv}")


if __name__ == "__main__":
    main()
