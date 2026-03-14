import os
import argparse
import jsonlines
from datasets import load_dataset


LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["premise", "hypothesis", "combined"],
        help="Which length criterion to use",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Where to save the jsonl training subset",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Total number of examples to select",
    )
    return parser.parse_args()


def build_messages(premise, hypothesis, label):
    label_text = LABEL_MAP[label]

    user_prompt = (
        "Decide if the hypothesis is implied by the premise ('entailment'), "
        "if the hypothesis contradicts the premise ('contradiction'), "
        "or neither ('neutral').\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}"
    )

    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": label_text},
    ]


def get_length_score(premise, hypothesis, mode):
    if mode == "premise":
        return len(premise.split())
    elif mode == "hypothesis":
        return len(hypothesis.split())
    elif mode == "combined":
        return len(premise.split()) + len(hypothesis.split())
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    args = get_args()

    print("Loading SNLI train split...")
    ds = load_dataset("snli", split="train")

    print("Filtering valid labels...")
    ds = ds.filter(lambda x: x["label"] in [0, 1, 2])

    per_class_counts = {
        0: args.num_samples // 3,
        1: args.num_samples // 3,
        2: args.num_samples - 2 * (args.num_samples // 3),
    }

    print("Scoring examples by length...")
    by_class = {0: [], 1: [], 2: []}

    for idx, ex in enumerate(ds):
        premise = ex["premise"]
        hypothesis = ex["hypothesis"]
        label = ex["label"]

        score = get_length_score(premise, hypothesis, args.mode)

        by_class[label].append({
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
            "score": score,
        })

        if (idx + 1) % 50000 == 0:
            print(f"Processed {idx + 1} examples")

    print("Sorting within each class...")
    selected = []

    for label in [0, 1, 2]:
        class_examples = sorted(by_class[label], key=lambda x: x["score"], reverse=True)
        top_k = per_class_counts[label]
        selected_class = class_examples[:top_k]
        selected.extend(selected_class)

        print(
            f"Label {label} ({LABEL_MAP[label]}): "
            f"selected {len(selected_class)} examples, "
            f"top score = {selected_class[0]['score'] if selected_class else 'N/A'}"
        )

    print(f"Total selected examples: {len(selected)}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    print(f"Saving to {args.output_file} ...")
    with jsonlines.open(args.output_file, mode="w") as writer:
        for ex in selected:
            messages = build_messages(
                premise=ex["premise"],
                hypothesis=ex["hypothesis"],
                label=ex["label"],
            )
            writer.write({"messages": messages})

    print("Done.")


if __name__ == "__main__":
    main()
