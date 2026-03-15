import os
import argparse
import jsonlines
import spacy
from datasets import load_dataset


LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Where to save the balanced parse-complexity subset",
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


def tree_depth(token):
    children = list(token.children)
    if not children:
        return 1
    return 1 + max(tree_depth(child) for child in children)


def sent_complexity(text, nlp):
    doc = nlp(text)

    roots = [tok for tok in doc if tok.head == tok]
    if not roots:
        depth = 1
    else:
        depth = max(tree_depth(root) for root in roots)

    num_tokens = len([tok for tok in doc if not tok.is_space])
    num_clauses = sum(1 for tok in doc if tok.dep_ in {"ccomp", "xcomp", "advcl", "relcl", "acl"})

    return {
        "depth": depth,
        "num_tokens": num_tokens,
        "num_clauses": num_clauses,
    }


def compute_complexity_score(premise, hypothesis, nlp):
    p = sent_complexity(premise, nlp)
    h = sent_complexity(hypothesis, nlp)

    score = (
        p["depth"] + h["depth"]
        + 0.2 * (p["num_tokens"] + h["num_tokens"])
        + 1.5 * (p["num_clauses"] + h["num_clauses"])
    )
    return score, p, h


def main():
    args = get_args()

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading SNLI train split...")
    ds = load_dataset("snli", split="train")

    print("Filtering valid labels...")
    ds = ds.filter(lambda x: x["label"] in [0, 1, 2])

    per_class_counts = {
        0: args.num_samples // 3,
        1: args.num_samples // 3,
        2: args.num_samples - 2 * (args.num_samples // 3),
    }

    print("Scoring examples by parse complexity...")
    by_class = {0: [], 1: [], 2: []}

    for idx, ex in enumerate(ds):
        premise = ex["premise"]
        hypothesis = ex["hypothesis"]
        label = ex["label"]

        score, p_stats, h_stats = compute_complexity_score(premise, hypothesis, nlp)

        by_class[label].append({
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
            "score": score,
            "premise_depth": p_stats["depth"],
            "hypothesis_depth": h_stats["depth"],
            "premise_len": p_stats["num_tokens"],
            "hypothesis_len": h_stats["num_tokens"],
            "premise_clauses": p_stats["num_clauses"],
            "hypothesis_clauses": h_stats["num_clauses"],
        })

        if (idx + 1) % 10000 == 0:
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
