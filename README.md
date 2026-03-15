# COM3610 Dissertation Project

### Project Title
Improving the Robustness of Fine-Tuned Large Language Models through Strategic Training Data Selection

### Project Overview
This project investigates how to strategically select a small subset of high-value training examples from a large-scale Natural Language Inference (NLI) dataset in order to fine-tune a Large Language Model (LLM) more effectively and improve its out-of-distribution (OOD) generalization and robustness.  
In standard fine-tuning pipelines, models often learn superficial patterns, annotation artefacts, and dataset-specific biases, which can lead to poor performance on unseen distributions. To address this issue, this project does not directly fine-tune on the full training set. Instead, it selects approximately 10,000 more representative, challenging, or diverse examples from around 500,000 SNLI training instances and uses them to perform parameter-efficient fine-tuning with LoRA.

### Project Objectives
- Reproduce and adapt an existing NLI evaluation pipeline
- Replace the original closed-source API-based model with an open-source Hugging Face model
- Compare different training data selection strategies
- Analyze how these strategies affect generalization on Standard-OOD and Challenge-OOD benchmarks
- Identify which data selection methods are most helpful for improving robustness

### Training Data Selection Strategies
This project considers and compares the following training data selection strategies:
1. Random Selection：Randomly sample 10,000 examples from the training set as a baseline.
2. Length-based Selection：Prefer longer input examples. This strategy assumes that longer examples tend to contain more semantic information and more complex linguistic structure.
3. Embedding Diversity Selection：Use a smaller model to compute embeddings for each example, then select examples that are more different from one another in order to maximize coverage and diversity.
4. Low-confidence Selection：Train a smaller model and use it to re-predict the training set. Select examples for which the model has low confidence, under the assumption that these examples are more challenging and more valuable for robust learning.
5. Parse Complexity Selection：Use syntactic parse trees or other linguistic complexity measures to prioritize structurally more complex examples.
6. Perplexity-based Selection：Use a language model to compute the perplexity of each example and prioritize examples that are harder for the language model to predict.
7. Possible Extensions：In addition to the strategies above, the project may explore new methods such as:
  - Joint diversity and difficulty scoring
  - Debiasing-based selection
  - Model disagreement-driven selection
  - Label-balanced difficulty stratification
  - Linguistic phenomenon coverage strategies

### Project Workflow
#### Step 1: Reproduction and Environment Setup
- Run the code provided by the reference paper
- Replace the original OpenAI API calls with a Hugging Face open-source model
- Configure the local or HPC environment

#### Step 2: Random Sampling Experiments
- Randomly select different training subsets
- Evaluate model performance on Standard-OOD and Challenge-OOD
- Observe variance caused by random sampling

#### Step 3: Implement a Simple Strategy
- Start with a simple and reproducible data selection method, such as length-based selection
- Use it as the first non-random strategy for comparison

#### Step 4: Compare Multiple Strategies
- Select around 3–4 data selection methods for full experiments
- Compare their performance across multiple test sets
- Analyze whether the performance gains are stable

#### Step 5: Summarize and Write the Report
- Summarize which types of methods work best
- Discuss the strengths and weaknesses of each strategy
- Propose possible improvements and future directions
