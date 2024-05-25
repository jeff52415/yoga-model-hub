import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)


def load_llm_model(
    model_id: str,
    quantization: bool = False,
    device: str = "auto",
    peft_model: bool = False,
    peft_folder: str = None,
):
    """
    Load a pre-trained LLM model and tokenizer from Hugging Face with optional quantization.

    Args:
        model_id (str): The model identifier from Hugging Face or the local path to the model.
        quantization (bool): Flag to indicate whether to use quantization. Default is False.
        device (str): The device to load the model on ('cpu', 'cuda', or 'auto'). Default is 'auto'.
        peft_model (bool): Flag to indicate whether to load a PEFT model. Default is False.
        peft_folder (str): The folder path to the PEFT model. Required if peft_model is True.

    Returns:
        model: The loaded LLM model.
        tokenizer: The loaded tokenizer.
    """
    if quantization:
        # Quantization specific imports and configuration
        # nf4 quantization is used for 4-bit quantization
        # bfloat16 is used for 4-bit compute dtype
        # double quantization can be used for better performance
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map=device,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)

    if peft_model:
        if peft_folder is None:
            raise ValueError("PEFT model folder path is required.")
        model = PeftModel.from_pretrained(model, peft_folder)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Set pad token to eos token for LLM models
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def save_model_local(model, tokenizer, save_path: str):
    """
    Save the LLM model and tokenizer to a local directory.

    Args:
        model: The LLM model to be saved.
        tokenizer: The tokenizer to be saved.
        save_path (str): The local directory path where the model and tokenizer will be saved.
    """
    model.save_pretrained(save_path, safe_serialization=False)
    tokenizer.save_pretrained(save_path)


# Example usage
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
quantization = True  # Set this to False if quantization is not needed
device = "auto"  # Set this to 'cpu' or 'cuda' if specific device is needed

model, tokenizer = load_llm_model(model_id, quantization, device)
save_path = "/tmp/llmav3/"
save_model_local(model, tokenizer, save_path)
