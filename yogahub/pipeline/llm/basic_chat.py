from langchain import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import GenerationConfig, pipeline

from yogahub.pipeline.llm.prompt_template import default_chat_template

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["user_query"],
    template=default_chat_template,
)


def default_pipeline(
    model, tokenizer, max_new_tokens=256, top_k=50, temperature=0.1, **generation_kwargs
):
    """
    Load the LLM model and tokenizer, create the HuggingFace pipeline, and generate a RunnableSequence.

    Args:
        model: The pre-trained model loaded from Hugging Face.
        tokenizer: The tokenizer loaded from Hugging Face.
        max_new_tokens (int): The maximum number of new tokens to generate. Default is 200.
        top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering. Default is 50.
        temperature (float): The value used to module the next token probabilities. Default is 0.1.
        **generation_kwargs: Additional keyword arguments to pass to the HuggingFace pipeline.

    Returns:
        extract_answer (str): The generated response from the model.
    """
    # reference: https://huggingface.co/docs/transformers/main_classes/text_generation

    terminators = [tokenizer.eos_token_id]

    # not stoping issue: https://github.com/ollama/ollama/issues/3759
    for stop in [
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|end_of_text|>",
    ]:
        terminators.append(tokenizer.convert_tokens_to_ids(stop))

    # Set pad token to eos token for LLM models
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_ids = tokenizer.pad_token_id

    config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_k,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        **generation_kwargs,
    )

    # Create the HuggingFace pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=config,
    )

    # Wrap the pipeline with HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=pipe)

    # Create the RunnableSequence
    sequence = prompt_template | llm
    # Generate the query based on the user request
    return sequence


# Define a function to generate responses based on user input
def generate_query(user_query, sequence, return_prompt=False):
    inputs = {
        "user_query": user_query,
    }
    prompt = prompt_template.format(**inputs)

    response = sequence.invoke(inputs)
    extract_answer = response[len(prompt) : len(response)]

    # Tentatively fix non-stoping issue
    if "How was my response?" in extract_answer:
        index = extract_answer.index("How was my response?")
        extract_answer = extract_answer[:index]
    if "In this response, you:" in extract_answer:
        index = extract_answer.index("In this response, you:")
        extract_answer = extract_answer[:index]

    if return_prompt:
        prompt, extract_answer
    return extract_answer


# Example usage
# sequence = default_pipeline(model, tokenizer)
# user_query = "Do I need any special equipment to practice yoga?"
# output = generate_query(user_query, sequence)
