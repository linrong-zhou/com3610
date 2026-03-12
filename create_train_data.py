import json
import random
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
import argparse
from utils import load_hf_dataset, save_data
from datasets import concatenate_datasets, Dataset, Sequence, ClassLabel, load_from_disk
import os
import torch
import transformers
import math

import numpy as np

def get_args():
    '''
    Parse command line arguments
    :return params:
    '''
    parser = argparse.ArgumentParser(
            description="Training model parameters")

    # Data locations
    parser.add_argument("--description", type=str, default="default",
                        help="Name of hf dataset")

    parser.add_argument("--split_name", type=str, default="default",
                        help="Split for hf dataset")

    parser.add_argument("--name_id", type=str, default="default",
                        help="Name to save data")

    parser.add_argument("--random_seed", type=int, default=13,
                        help="random seed for sampling")

    # Sampling
    parser.add_argument("--num_samples_firstsplit", type=int, default=10000, # Our baseline starts with 10k examples
                        help="Size of the initial training data.")

    parser.add_argument("--num_samples_nextsplit", type=int, default=10000, # We consider another 10k examples for possible replacement candidates
                        help="Size of the extra data where we look for good examples to drop into original training data")

    parser.add_argument("--method_name", type=str, default='random',
                        help="Method name")
    
    parser.add_argument("--sample_file_firstsplit", type=str, default="default",
                        help="File to load for misclassified sampling prediction")

    parser.add_argument("--sample_file_nextsplit", type=str, default="default",
                        help="File to load misclassified sampling prediction for next split")

    parser.add_argument("--do_replacement", type=int, default=0,
                        help="If we do sampling, or leave the training data as is")
    
    # How many examples we want to change across each class
    parser.add_argument("--examples_to_upsample_by_class", type=int, default=500, # K = 5% (see Section 4 of the paper)
                        help="How many examples to re-select by class if do_replacement is True")

    # Saving
    parser.add_argument("--upload_to_openai", type=str, default="default",
                        help="'yes' to upload to openai, 'no' otherwise")

    # Extra arguments for difficulty_score method
    parser.add_argument("--difficulty_score_location", type=str, default="default",
                        help="gpt")

    parser.add_argument("--concat_data_location", type=str, default="default",
                        help="File to load concatenated predictions")

    # Extra arguments for unlabelled data generation
    parser.add_argument("--unlab_data_location", type=str, default="default",
                        help="File to load unlabelled data")
    
    params, _ = parser.parse_known_args()

    return params

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

def create_user_prompt(obs, params, inference=False):


    prompt_to_user = "Decide if the hypothesis is implied by the premise ('entailment'), if the hypothesis contradicts the premise ('contradiction'), or neither ('neutral')."
    prompt_to_user += "\nPremise: " + obs['premise']
    prompt_to_user += '\nHypothesis: ' + obs['hypothesis']

    return prompt_to_user

def create_ft_data(user_prompt, obs, params, inference=False):

    assert obs['label'] in [0, 1, 2], str(obs['label'])

    messages = []
    messages.append({
            "role": "user",
            "content": user_prompt})

    if inference:
        return messages

    else:

        if obs['label'] == 0:
            label = 'entailment'
        elif obs['label'] ==1:
            label = 'neutral'
        elif obs['label'] == 2:
            label = 'contradiction'

        messages.append({
                "role": "assistant",
                "content": label})

    return messages

def format_data(train_data, params, inference=False):

    all_data = []

    for i, obs in enumerate(train_data):

        if obs['label'] in  [0, 1, 2]:

            # Do not train on instances with an invalid label
            user_prompt = create_user_prompt(obs, params)
            messages = create_ft_data(user_prompt, obs, params, inference=inference)

            # If doing inference, we include the label in the dataset so we can do evaluation
            if inference:
                all_data.append({"messages": messages, 'label': obs['label']})
            else:
                all_data.append({"messages": messages})

    return all_data


def split_difficulty_score_response_method(train_data_split, params, firstsplit=True, num_selected_by_class=None):

    if firstsplit:
        num_obs = params.num_samples_firstsplit
    else:
        num_obs = params.num_samples_nextsplit

    with open(params.difficulty_score_location, "r") as f:
        data = json.load(f)
        data = data['run_1_snli_train']

    if firstsplit:
        assert num_selected_by_class
        num_desirable_by_class = {
                '0': len(train_data_split.filter(lambda example: example['label'] == 0)) - num_selected_by_class['0'],
                '1': len(train_data_split.filter(lambda example: example['label'] == 1)) - num_selected_by_class['1'],
                '2': len(train_data_split.filter(lambda example: example['label'] == 2)) - num_selected_by_class['2']
                }
    else:
        num_desirable_by_class = {
                '0': params.examples_to_upsample_by_class,
                '1': params.examples_to_upsample_by_class,
                '2': params.examples_to_upsample_by_class
                }

    desirable_ids = []
    undesirable_ids = []

    all_scores = {'0': [], '1': [], '2': []}

    for idx in range(params.num_samples_firstsplit + params.num_samples_nextsplit):

        if not firstsplit:
            idx_in_train_split = idx-params.num_samples_firstsplit
        else:
            idx_in_train_split = idx

        if (not firstsplit and idx >= params.num_samples_firstsplit and idx < params.num_samples_firstsplit + params.num_samples_nextsplit) \
                or (firstsplit and idx < params.num_samples_firstsplit):

            lab = train_data_split[idx_in_train_split]['label']
            response_dict = parse_response(data[str(idx)]['full_response'])

            # We consider parsable responses. Non-parsable ones are considered non-desirable
            if response_dict != {}:
                score = response_dict['difficulty']
                all_scores[str(lab)].append((idx_in_train_split, score))

    # Sorts the predictions by combined scores
    all_scores['0'].sort(key=lambda tup: tup[1], reverse=True)
    all_scores['1'].sort(key=lambda tup: tup[1], reverse=True)
    all_scores['2'].sort(key=lambda tup: tup[1], reverse=True)

    desirable_ids = []

    for class_lab in all_scores:
        for idx_score_tup in all_scores[class_lab][:num_desirable_by_class[class_lab]]:
            idx, score = idx_score_tup
            desirable_ids.append(idx)

            if not firstsplit:
                idx_in_difficulty_score_dict = idx+params.num_samples_firstsplit
            else:
                idx_in_difficulty_score_dict = idx

    for idx in range(num_obs):
        if idx not in desirable_ids:
            undesirable_ids.append(idx)

    return desirable_ids, undesirable_ids


def split_misclassified_sampling_baseline_method(train_data_split, preds_on_baseline_examples, params, firstsplit=True, num_selected_by_class=None):

    num_obs = params.num_samples_nextsplit

    # First we check that we only have a single soft prediction on the training data
    assert len(preds_on_baseline_examples) == 1
    pred_dict = preds_on_baseline_examples[list(preds_on_baseline_examples.keys())[0]]

    desirable_ids = []
    undesirable_ids = []

    incorrect_idx = {'0': [], '1': [], '2': []}

    # We rank the uncertainty of observations for each class
    for idx in range(num_obs):
        num_correct = 0

        idx_pred_dict = pred_dict[str(idx)]

        correct_pred = False

        if idx_pred_dict['label'] == 1:
            if idx_pred_dict['full_response'] == 'neutral':
                correct_pred = True

        elif idx_pred_dict['label'] == 0:
            if idx_pred_dict['full_response'] == 'entailment':
                correct_pred = True

        elif idx_pred_dict['label'] == 2:
            if idx_pred_dict['full_response'] == 'contradiction':
                correct_pred = True

        if not correct_pred:
            incorrect_idx[str(idx_pred_dict['label'])].append((idx))

    max_num_desirable_by_class = {
            '0': params.examples_to_upsample_by_class,
            '1': params.examples_to_upsample_by_class,
            '2': params.examples_to_upsample_by_class
            }

    desirable_ids = []

    for class_lab in incorrect_idx:
        for idx in incorrect_idx[class_lab][:max_num_desirable_by_class[class_lab]]:
            desirable_ids.append(idx)

    for idx in range(num_obs):
        if idx not in desirable_ids:
            undesirable_ids.append(idx)

    return desirable_ids, undesirable_ids


def split_softpreds_mostuncertain_method(train_data_split, preds_on_baseline_examples, params, firstsplit=True, num_selected_by_class=None):

    """
    We choose the most uncertain examples as 'desirable', from the specified train split
    .. the most certain examples are therefore 'undesirable'
    """

    if firstsplit:
        num_obs = params.num_samples_firstsplit
    else:
        num_obs = params.num_samples_nextsplit

    # First we check that we only have a single soft prediction on the training data
    assert len(preds_on_baseline_examples) == 1
    pred_dict = preds_on_baseline_examples[list(preds_on_baseline_examples.keys())[0]]

    desirable_ids = []
    undesirable_ids = []

    # We calculate the uncertainty by class, as all our sampling should not change the class balance
    all_scores = {'0': [], '1': [], '2': []}

    # We rank the uncertainty of observations for each class
    for idx in range(num_obs):
        num_correct = 0
        idx_scores = []

        idx_pred_dict = pred_dict[str(idx)]

        distribution = [
                idx_pred_dict['soft_probs']['ent'], 
                idx_pred_dict['soft_probs']['neutral'], 
                idx_pred_dict['soft_probs']['contr']
                ]

        score = 0
        for prob in distribution:
            if prob > 0:
                score += -1 * prob * math.log2(prob)

        all_scores[str(idx_pred_dict['label'])].append((idx, score))

    if firstsplit:
        assert num_selected_by_class
        num_desirable_by_class = {
                '0': len(train_data_split.filter(lambda example: example['label'] == 0)) - num_selected_by_class['0'],
                '1': len(train_data_split.filter(lambda example: example['label'] == 1)) - num_selected_by_class['1'],
                '2': len(train_data_split.filter(lambda example: example['label'] == 2)) - num_selected_by_class['2']
                }
    else:
        num_desirable_by_class = {
                '0': params.examples_to_upsample_by_class,
                '1': params.examples_to_upsample_by_class,
                '2': params.examples_to_upsample_by_class
                }
                
    # Sorts the predictions by uncertainty
    all_scores['0'].sort(key=lambda tup: tup[1], reverse=True)
    all_scores['1'].sort(key=lambda tup: tup[1], reverse=True)
    all_scores['2'].sort(key=lambda tup: tup[1], reverse=True)

    desirable_ids = []
    
    for class_lab in all_scores:
        for idx_score_tup in all_scores[class_lab][:num_desirable_by_class[class_lab]]:
            idx, score = idx_score_tup
            desirable_ids.append(idx)

    for idx in range(num_obs):
        if idx not in desirable_ids:
            undesirable_ids.append(idx)

    return desirable_ids, undesirable_ids


def random_selection(train_data_split, params, firstsplit=True, num_selected_by_class=None):

    if firstsplit:
        assert num_selected_by_class
        num_desirable_by_class = {
                '0': len(train_data_split.filter(lambda example: example['label'] == 0)) - num_selected_by_class['0'],
                '1': len(train_data_split.filter(lambda example: example['label'] == 1)) - num_selected_by_class['1'],
                '2': len(train_data_split.filter(lambda example: example['label'] == 2)) - num_selected_by_class['2']
                }
    else:
        num_desirable_by_class = {
                '0': params.examples_to_upsample_by_class,
                '1': params.examples_to_upsample_by_class,
                '2': params.examples_to_upsample_by_class
                }

    # We want to drop out params.examples_to_upsample_by_class from each class (so this number is replaced exactly)
    class_idx = {0: [], 1: [], 2: []}

    for idx, obs in enumerate(train_data_split):
        class_idx[obs['label']].append(idx)

    desirable_ids = []
    for class_lab in class_idx:
        desirable_ids += random.sample(class_idx[class_lab], num_desirable_by_class[str(class_lab)])


    undesirable_ids = [x for x in range(len(train_data_split)) if x not in desirable_ids]

    return desirable_ids, undesirable_ids


def split_into_desirable_and_undesirable(
        train_data_split,
        preds_on_baseline_examples,
        strategy,
        firstsplit=True,
        num_selected_by_class=None):

    if strategy == 'uncertainty_sampling':
        desirable_ids, undesirable_ids = split_softpreds_mostuncertain_method(
                train_data_split,
                preds_on_baseline_examples, 
                params, 
                firstsplit,
                num_selected_by_class)

    elif strategy == 'misclassified_sampling':
        desirable_ids, undesirable_ids = split_misclassified_sampling_baseline_method(
                train_data_split,
                preds_on_baseline_examples,
                params,
                firstsplit,
                num_selected_by_class)

    elif strategy == 'difficulty_score':
        desirable_ids, undesirable_ids = split_difficulty_score_response_method(
                train_data_split, 
                params, 
                firstsplit, 
                num_selected_by_class)

    elif strategy == 'random' or strategy == 'concat' or strategy == 'unlab':
        desirable_ids, undesirable_ids = random_selection(
                train_data_split, 
                params, 
                firstsplit, 
                num_selected_by_class)

    return desirable_ids, undesirable_ids


def count_by_class_in_desirables(train_data_split, desirable_ids, firstsplit, params):

    num_selected_by_class = {'0': 0, '1': 0, '2': 0}

    for idx, obs in enumerate(train_data_split):

        if idx in desirable_ids:
            num_selected_by_class[str(obs['label'])] += 1

    return num_selected_by_class

def get_desirable_and_undesirable_splits(train_data_split, strategy, params, run_num, firstsplit=True, num_selected_by_class=None):

    preds_on_baseline_examples = {}
    if firstsplit:
        misclassified_sampling_pred_dir = "saved_baseline_preds/" + params.sample_file_firstsplit
    else:
        misclassified_sampling_pred_dir = "saved_baseline_preds/" + params.sample_file_nextsplit

    if strategy in ['random', 'difficulty_score', 'concat', 'unlab']:
        preds_on_baseline_examples = None
    else:
        for json_file in os.listdir(misclassified_sampling_pred_dir):
            if ".json" in json_file:
                with open(misclassified_sampling_pred_dir + "/" + json_file) as f:
                    preds_on_baseline_examples[json_file.replace(".json", "")] = json.load(f)

    desirable_ids, undesirable_ids = split_into_desirable_and_undesirable(
            train_data_split,
            preds_on_baseline_examples, 
            strategy, 
            firstsplit,
            num_selected_by_class)

    num_selected_by_class = count_by_class_in_desirables(train_data_split, desirable_ids, firstsplit, params)

    splits = {'desirable': desirable_ids, 'undesirable': undesirable_ids}

    return splits, num_selected_by_class

def perform_all_sampling(train_data, params, run_num):

    train_data_firstsplit = train_data.filter(
            lambda example, idx: idx < params.num_samples_firstsplit, with_indices=True)

    train_data_nextsplit = train_data.filter(
            lambda example, idx: idx >= params.num_samples_firstsplit and idx < params.num_samples_firstsplit + params.num_samples_nextsplit, with_indices=True)

    # If we instead use the concatenated hypotheses or the generated data:
    if params.method_name == 'concat':
        train_data_nextsplit = load_from_disk("concat_data/" + params.concat_data_location + '.hf')
        train_data_nextsplit = train_data_nextsplit.filter(lambda example: example['label'] in [0, 1, 2])
        label_type = ClassLabel(names=['entailment', 'neutral', 'contradiction'])
        train_data_nextsplit = train_data_nextsplit.cast_column("label", label_type)

    elif params.method_name == 'unlab':
        train_data_nextsplit = load_from_disk('labelled_unlabelled_data/' + params.unlab_data_location)
        train_data_nextsplit = train_data_nextsplit.filter(lambda example: example['label'] in [0, 1, 2])
        label_type = ClassLabel(names=['entailment', 'neutral', 'contradiction'])
        train_data_nextsplit = train_data_nextsplit.cast_column("label", label_type)

    next_splits, num_selected_by_class = get_desirable_and_undesirable_splits(
            train_data_nextsplit,
            params.method_name,
            params,
            run_num,
            firstsplit=False,
            num_selected_by_class=None)

    first_splits, _ = get_desirable_and_undesirable_splits(
            train_data_firstsplit, 
            'random',
            params, 
            run_num, 
            firstsplit=True,
            num_selected_by_class=num_selected_by_class)

    train_data_firstsplit_kept = train_data_firstsplit.filter(
            lambda example, idx: idx in first_splits['desirable'], with_indices=True)

    train_data_nextsplit_kept = train_data_nextsplit.filter(
            lambda example, idx: idx in next_splits['desirable'], with_indices=True)

    train_data_new = concatenate_datasets([train_data_firstsplit_kept, train_data_nextsplit_kept])

   
    for class_lab in [0, 1, 2]:
        orig_examples_with_lab = train_data_firstsplit.filter(lambda example: example['label'] == class_lab)
        new_examples_with_lab = train_data_new.filter(lambda example: example['label'] == class_lab)

    return train_data_new


def create_train_data(params, run_num=0, aug_data_fine_tuned_results=None):

    # Load the training data
    train_data = load_hf_dataset(params.description, params.split_name, shuffle_dataset=True)
  
    train_data = train_data.filter(lambda example: example['label'] in [0, 1, 2])

    # If we are upsampling
    if params.do_replacement:
        train_data = perform_all_sampling(train_data, params, run_num)
    else:
        train_data = train_data.filter(lambda example, idx: idx < params.num_samples_firstsplit, with_indices=True)

    # Formatting our training data with our prompts
    all_data = format_data(train_data, params)

    # Shuffle
    set_seeds(params.random_seed)
    random.shuffle(all_data)

    return all_data


def parse_response(response):

   response = response.strip().strip("\n").strip()
   response_list = response.split(":")
   valid = True

   if not (response_list[0] == 'Correctness'):
       valid = False

   if valid:
       if not ('\nDifficulty' in response_list[1]):
           valid = False
       if valid:
           correctness = response_list[1].replace("\nDifficulty", "").replace(" ", "")
           correctness = int(correctness)

   if valid:
       if not('\nFluency' in response_list[2]):
           valid = False
       if valid:
           difficulty = response_list[2].replace("\nFluency", "").replace(" ", "")
           difficulty = int(difficulty)

   if valid:
       if not ('\nPlausibility' in response_list[3]):
           valid = False
       if valid:
           fluency = response_list[3].replace("\nPlausibility", "").replace(" ", "")
           fluency = int(fluency)

   if valid:
       if 'a' in response_list[4]:
           valid = False
       else:
           plausibility = response_list[4].replace(" ", "")
           plausibility = int(plausibility)

   if valid:
       return {
               'correctness': correctness,
               'difficulty': difficulty,
               'fluency': fluency,
               'plausibility': plausibility}
   
   return {}



if __name__ == "__main__":

    params = get_args()

    # Set Boolean params
    params.do_replacement = bool(params.do_replacement)

    # Check params are valid
    print(params)

    assert params.upload_to_openai in ['yes', 'no'], "upload_to_openai must be 'yes' or 'no'"

    # Creating our training data
    all_data = create_train_data(params)

    print("Final training data after sampling:", len(all_data))

    # Saving our training data
    file_name = params.name_id \

    print("File name:", file_name)

    assert not os.path.isfile(
            "saved_data_for_training/" + file_name + ".jsonl"), \
                    "Output filename exists, try changing name_id: " + file_name
    save_data(all_data, "saved_data_for_training/" + file_name + ".jsonl")

    # Uploading our training data to openai
    data_path = "saved_data_for_training/" + file_name + ".jsonl"

    if params.upload_to_openai == 'yes':
        if OpenAI is None:
            raise ImportError("openai package is not installed, but upload_to_openai='yes' was requested.")
        client = OpenAI()

        client.files.create(
            file=open(data_path, "rb"),
            purpose="fine-tune"
        )


