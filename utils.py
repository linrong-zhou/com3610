import random
import os
import json
import datasets

path = os.path.abspath(os.getcwd())

import transformers
from transformers import (
    set_seed,
)

import torch
import jsonlines
from datasets import load_dataset, load_from_disk, Dataset, ClassLabel

def load_hf_dataset(description, split_name, from_file=False, aug_data_location=None, shuffle_dataset=False):

    # Load dataset
    print("Description:", description)
    print("Split name:", split_name)

    if description == 'inli':

        dataset = load_dataset("csv", data_files="data/inli_test.csv", split='train')
        
        full_dataset_hyp = []
        full_dataset_prem = []
        full_dataset_label = []

        if split_name == 'nli':

            for obs_no in range(len(dataset)):
                full_dataset_prem.append(dataset['premise'][obs_no])
                full_dataset_hyp.append(dataset['explicit_entailment'][obs_no])
                full_dataset_label.append(0)

                full_dataset_prem.append(dataset['premise'][obs_no])
                full_dataset_hyp.append(dataset['neutral'][obs_no])
                full_dataset_label.append(1)

                full_dataset_prem.append(dataset['premise'][obs_no])
                full_dataset_hyp.append(dataset['contradiction'][obs_no])
                full_dataset_label.append(2)
                

        elif split_name == 'implied':

            for obs in dataset:
                full_dataset_prem.append(obs['premise'])
                full_dataset_hyp.append(obs['implied_entailment'])
                full_dataset_label.append(0)


        loaded_data = {'premise': full_dataset_prem, 'hypothesis': full_dataset_hyp, 'label': full_dataset_label}
        loaded_data = Dataset.from_dict(loaded_data)
        loaded_data = loaded_data.filter(lambda example: example['label'] in [0, 1, 2])

        return loaded_data

    elif description == 'copa_nli':
 
        assert split_name == 'test'

        loaded_data = load_dataset(
                'pkavumba/balanced-copa', split='test'
        )

        full_dataset_hyp = []
        full_dataset_prem = []
        full_dataset_label = []

        for obs in loaded_data:

            if obs['question'] == 'cause':
                full_dataset_prem.append(obs['premise'])
                hyp_text = '"' + obs['choice1'] + '" is a more likely cause of this than "' + obs['choice2'] + '"'
                full_dataset_hyp.append(hyp_text)
                assert obs['label'] in [0,1]
                if obs['label'] == 0:
                    full_dataset_label.append(0)
                elif obs['label'] == 1:
                    full_dataset_label.append(1)

                full_dataset_prem.append(obs['premise'])
                hyp_text = '"' + obs['choice2'] + '" is a more likely cause of this than "' + obs['choice1'] + '"'
                full_dataset_hyp.append(hyp_text)
                assert obs['label'] in [0,1]
                if obs['label'] == 0:
                    full_dataset_label.append(1)
                elif obs['label'] == 1:
                    full_dataset_label.append(0)

            elif obs['question'] == 'effect':

                full_dataset_prem.append(obs['premise'])
                hyp_text = '"' + obs['choice1'] + '" is a more likely effect of this than "' + obs['choice2'] + '"'
                full_dataset_hyp.append(hyp_text)
                assert obs['label'] in [0,1]
                if obs['label'] == 0:
                    full_dataset_label.append(0)
                elif obs['label'] == 1:
                    full_dataset_label.append(1)

                full_dataset_prem.append(obs['premise'])
                hyp_text = '"' + obs['choice2'] + '" is a more likely effect of this than "' + obs['choice1'] + '"'
                full_dataset_hyp.append(hyp_text)
                assert obs['label'] in [0,1]
                if obs['label'] == 0:
                    full_dataset_label.append(1)
                elif obs['label'] == 1:
                    full_dataset_label.append(0)

        loaded_data = {'premise': full_dataset_prem, 'hypothesis': full_dataset_hyp, 'label': full_dataset_label}
        loaded_data = Dataset.from_dict(loaded_data)
        loaded_data = loaded_data.filter(lambda example: example['label'] in [0, 1, 2])

        return loaded_data

    elif description == 'pietrolesci/nli_fever':

        loaded_data = load_dataset(
                description, split=split_name
        )

        # The hypothesis and premise are the wrong way around on nli_fever on huggingface
        loaded_data = loaded_data.rename_column("premise", "new_hypothesis")
        loaded_data = loaded_data.rename_column("hypothesis", "premise")
        loaded_data = loaded_data.rename_column("new_hypothesis", "hypothesis")

        return loaded_data

    elif description == 'alisawuffles/WANLI':

        loaded_data = load_dataset(
                description, split=split_name
        )

        formatted_label = []
        for obs in loaded_data:
            if obs['gold'] == 'entailment':
                formatted_label.append(0)
            elif obs['gold'] == 'neutral':
                formatted_label.append(1)
            elif obs['gold'] == 'contradiction':
                formatted_label.append(2)

            assert obs['gold'] in ['entailment', 'neutral', 'contradiction'], str(obs['gold'])

        loaded_data = loaded_data.add_column("label", formatted_label)

        return loaded_data

    elif description == 'allenai/scitail':
        loaded_data = load_dataset(
                description, 'snli_format', split=split_name
        )

        formatted_label = []
        for obs in loaded_data:
            if obs['gold_label'] == 'entails' or obs['gold_label'] == 'entailment':
                formatted_label.append(0)
            elif obs['gold_label'] == 'neutral':
                formatted_label.append(1)
            assert obs['gold_label'] in ['entails', 'neutral', 'entailment'], str(obs['gold_label'])

        loaded_data = loaded_data.add_column("label", formatted_label)

        loaded_data = loaded_data.rename_column("sentence1", "premise")
        loaded_data = loaded_data.rename_column("sentence2", "hypothesis")

        return loaded_data

    else:

        loaded_data = load_dataset(
                description, split=split_name
        )

        if shuffle_dataset:
            set_seeds(13)
            loaded_data = loaded_data.shuffle()

    return loaded_data

def set_seeds(seed_value: int) -> None:
    """
    Set seed for reproducibility.

    Args:
        seed_value: chosen random seed
    """
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    transformers.set_seed(seed_value)


def scitail_label_processing(example):
    if example['label'] == 'entails':
        example['label'] = 0
    elif example['label'] == 'neutral':
        example['label'] = 1

    return example


def save_data(data, name):
    """
    Saving dataset with given file name
    """
    with jsonlines.open(name, mode="w") as writer:
        writer.write_all(data)
        writer.close()


def is_pred_correct(llm_pred, label, dataset_desc):

    pred = llm_pred.strip().lower()

    if label == 0 and pred == "entailment":
        return True
    elif label == 1 and pred == "neutral":
        return True
    elif label == 2 and pred == "contradiction":
        return True
    elif str(label).lower() == pred:
        return True

    return False


