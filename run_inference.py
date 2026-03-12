import os
import json
import argparse
import random
import numpy as np
import torch
import datasets
import transformers

from create_train_data import format_data
from utils import load_hf_dataset
from peft import PeftModel

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


HF_MODEL_NAME = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}

_MODEL = None
_TOKENIZER = None


def get_args():
    parser = argparse.ArgumentParser(description="Inference parameters")

    parser.add_argument("--num_samples_firstsplit", type=int, default=10000, help="size of firstsplit")
    parser.add_argument("--num_samples_nextsplit", type=int, default=10000, help="size of nextsplit")
    parser.add_argument("--model_id", type=str, default="default", help="openai model id or local LoRA adapter path")
    parser.add_argument("--model_type", type=str, default="default", help="gpt4mini / mistral / llama")
    parser.add_argument("--name_id", type=str, default="default", help="text for saving results")
    parser.add_argument("--temperature", type=float, default=0.00001, help="temperature for LLM calls")
    parser.add_argument("--top_p", type=float, default=0.00001, help="top p for LLM calls")
    parser.add_argument("--random_seed", type=int, default=13, help="random seed")
    parser.add_argument("--num_runs", type=int, default=1, help="number of runs")
    parser.add_argument(
        "--make_predictions_on_training_data",
        type=int,
        default=0,
        help="whether making predictions on training data",
    )
    parser.add_argument(
        "--training_data_to_make_preds_on",
        type=str,
        default="default",
        help="name of training data if making training predictions",
    )
    parser.add_argument("--aug_data", type=str, default="default", help="unlabelled dataset location")
    parser.add_argument("--unlabelled", type=int, default=0, help="whether predicting on unlabelled data")
    parser.add_argument("--filter_type", type=str, default="default", help="nextsplit / default")
    parser.add_argument("--gpt_soft_probs", type=int, default=0, help="whether GPT returns soft probs")

    params, _ = parser.parse_known_args()
    return params


def set_seeds(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    transformers.set_seed(seed_value)


def load_hf_model(params):
    global _MODEL, _TOKENIZER

    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    base_model_name = HF_MODEL_NAME[params.model_type]

    if params.model_id == "local":
        print(f"Loading base HF model: {base_model_name}")

        _TOKENIZER = transformers.AutoTokenizer.from_pretrained(
            base_model_name,
            use_fast=False,
        )
        if _TOKENIZER.pad_token is None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token

        if torch.cuda.is_available():
            _MODEL = transformers.AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            _MODEL = transformers.AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
            )
            _MODEL = _MODEL.to("cpu")

    else:
        adapter_dir = params.model_id
        print(f"Loading base HF model: {base_model_name}")
        print(f"Loading LoRA adapter from: {adapter_dir}")

        _TOKENIZER = transformers.AutoTokenizer.from_pretrained(
            adapter_dir,
            use_fast=False,
        )
        if _TOKENIZER.pad_token is None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token

        if torch.cuda.is_available():
            base_model = transformers.AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            base_model = transformers.AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
            )
            base_model = base_model.to("cpu")

        _MODEL = PeftModel.from_pretrained(base_model, adapter_dir)

    _MODEL.eval()
    print("HF model loaded.")
    return _MODEL, _TOKENIZER


def messages_to_prompt(messages):
    parts = []
    for m in messages:
        role = m["role"].strip().lower()
        content = m["content"].strip()

        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")

    return "\n\n".join(parts)


def build_strict_nli_prompt(obs):
    base_prompt = messages_to_prompt(obs["messages"])

    prompt = (
        "You are an NLI classifier.\n"
        "Given a premise and a hypothesis, answer with exactly one word only.\n"
        "Valid labels are:\n"
        "entailment\n"
        "neutral\n"
        "contradiction\n\n"
        "Do not explain your answer.\n"
        "Do not write a sentence.\n"
        "Output exactly one label.\n\n"
        f"{base_prompt}\n\n"
        "Answer:"
    )
    return prompt


def _gpt_call(obs, params):
    if OpenAI is None:
        raise ImportError("openai package is not installed, but model_type='gpt4mini' was requested.")

    client = OpenAI()
    model = params.model_id

    if not params.gpt_soft_probs:
        response = client.chat.completions.create(
            temperature=params.temperature,
            top_p=params.top_p,
            model=model,
            messages=obs["messages"],
            logprobs=False,
        )
        soft_probs = {}
    else:
        response = client.chat.completions.create(
            temperature=params.temperature,
            top_p=params.top_p,
            model=model,
            messages=obs["messages"],
            logprobs=True,
            top_logprobs=10,
        )
        soft_probs = {}

    full_response = response.choices[0].message.content
    return full_response, soft_probs


def _hf_call(obs, params):
    model, tokenizer = load_hf_model(params)

    prompt = build_strict_nli_prompt(obs)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    soft_probs = {}
    return text, soft_probs


def normalize_response(response):
    text = response.strip().lower()

    if "contradiction" in text or "contradict" in text:
        return "contradiction"
    if "neutral" in text:
        return "neutral"
    if (
        "entailment" in text
        or "entails" in text
        or "entailed" in text
        or "implied" in text
        or "implies" in text
        or "imply" in text
    ):
        return "entailment"

    text = text.replace(".", " ").replace(",", " ").replace(":", " ").replace(";", " ")
    tokens = text.split()

    for tok in tokens:
        if tok in ["contradiction", "contradict"]:
            return "contradiction"
        if tok == "neutral":
            return "neutral"
        if tok in ["entailment", "entails", "implied", "implies"]:
            return "entailment"

    return response.strip()


def get_pred(dataset, params, run_num):
    set_seeds(params.random_seed + run_num)
    output_dict = {}

    for i, obs in enumerate(dataset):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Processing example {i + 1}/{len(dataset)}")

        no_tries = 0
        response_valid = False
        max_no_tries = 1
        soft_probs = {}
        response = ""

        while not response_valid and no_tries < max_no_tries:
            if params.model_type == "gpt4mini":
                response, soft_probs = _gpt_call(obs, params)
            elif params.model_type in ["mistral", "llama"]:
                response, soft_probs = _hf_call(obs, params)
            else:
                raise ValueError(f"Unsupported model_type: {params.model_type}")

            response = normalize_response(response)

            if response.lower() in ["neutral", "contradiction", "entailment"]:
                response_valid = True
            else:
                response_valid = False

            no_tries += 1

        if params.model_type == "gpt4mini" and params.gpt_soft_probs:
            output_dict[i] = {
                "label": obs["label"],
                "full_response": response,
                "response_valid": response_valid,
                "soft_probs": soft_probs,
            }
        else:
            output_dict[i] = {
                "label": obs["label"],
                "full_response": response,
                "response_valid": response_valid,
            }

    return output_dict


if __name__ == "__main__":
    params = get_args()

    params.make_predictions_on_training_data = bool(params.make_predictions_on_training_data)
    params.unlabelled = bool(params.unlabelled)
    params.gpt_soft_probs = bool(params.gpt_soft_probs)

    if params.make_predictions_on_training_data:
        assert params.filter_type in ["nextsplit"]

    if params.filter_type in ["nextsplit"]:
        assert params.make_predictions_on_training_data

    print(params)

    assert params.name_id != "default"
    assert 0 <= params.temperature <= 2
    assert 0 <= params.top_p <= 1
    assert params.model_type in ["gpt4mini", "mistral", "llama"]

    # 先去掉 inli，因为你当前缺本地 csv 文件
    all_data_tuples = [
        ("snli", "test"),
        ("snli", "validation"),
        ("multi_nli", "validation_matched"),
        ("multi_nli", "validation_mismatched"),
        ("anli", "test_r1"),
        ("anli", "test_r2"),
        ("anli", "test_r3"),
        ("allenai/scitail", "test"),
        ("pietrolesci/nli_fever", "dev"),
        ("copa_nli", "test"),
        ("alisawuffles/WANLI", "test"),
    ]

    if params.training_data_to_make_preds_on != "default":
        assert params.make_predictions_on_training_data

    if params.make_predictions_on_training_data:
        assert params.training_data_to_make_preds_on != "default"
        assert not params.unlabelled
        all_data_tuples = [(params.training_data_to_make_preds_on, "train")]
    elif params.unlabelled:
        assert params.aug_data != "default"
        all_data_tuples = [("unlabelled", "train")]

    for data_tuple in all_data_tuples:
        for run in range(params.num_runs):
            run += 1
            data_desc, data_split = data_tuple

            print(f"Description: {data_desc}")
            print(f"Split name: {data_split}")

            exp_name = "run_" + str(run) + "_" + data_desc + "_" + data_split
            exp_name = exp_name.replace("/", "_")

            if params.make_predictions_on_training_data:
                output_file = "saved_baseline_preds/" + params.name_id
            elif params.unlabelled:
                output_file = "unlabelled_data_predictions/" + params.name_id
            else:
                output_file = "inference_outputs/" + params.name_id

            if not os.path.exists(output_file):
                os.makedirs(output_file)

            save_path = output_file + "/" + exp_name + ".json"

            if os.path.exists(save_path):
                print(f"Skipping existing file: {save_path}")
                continue

            loaded_dataset = load_hf_dataset(
                data_desc,
                data_split,
                aug_data_location=params.aug_data,
                shuffle_dataset=True,
            )
            print("dataset loaded from huggingface")

            if params.make_predictions_on_training_data:
                loaded_dataset = loaded_dataset.filter(lambda example: example["label"] in [0, 1, 2])

                print("Size before filtering:", len(loaded_dataset))
                loaded_dataset = loaded_dataset.filter(
                    lambda example, index: index < params.num_samples_nextsplit + params.num_samples_firstsplit
                    and index >= params.num_samples_firstsplit,
                    with_indices=True,
                )
                print("Size after filtering:", len(loaded_dataset))

            loaded_dataset = format_data(loaded_dataset, params, inference=True)
            loaded_dataset = datasets.Dataset.from_list(loaded_dataset)

            output_dict = get_pred(loaded_dataset, params, run)

            with open(save_path, "w") as f:
                json.dump(output_dict, f, indent=2)

            print(f"Saved output to: {save_path}")
