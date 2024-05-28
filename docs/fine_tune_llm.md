# Fine-Tuning Large Language Models (LLM) Guide

## Installation

### Step 1: Install the package
To begin, clone the repository and navigate to the root directory of the project. Then, run the following command to install the package with the necessary dependencies for large language models (LLM):

```bash
pip install -e ."[llm]"
```

### Step 2: Install Torchtune
The latest official release of `torchtune` is outdated (as of 2024/05/28). Therefore, you need to manually install the latest version directly from the GitHub repository:

```bash
pip install git+https://github.com/pytorch/torchtune.git@main
```

## Fine-Tuning Guide

For step-by-step guidance on how to fine-tune a large language model, refer to the Jupyter notebook provided in the repository:

- [notebooks/fine_tune_llm.ipynb](../notebooks/fine_tune_llm.ipynb)

This notebook contains detailed instructions and code examples to help you through the fine-tuning process.
