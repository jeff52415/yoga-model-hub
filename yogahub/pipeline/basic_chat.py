from langchain import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from yogahub.pipeline.prompt_template import default_chat_template

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["user_query"],
    template=default_chat_template,
)


def default_pipeline(
    model, tokenizer, max_new_tokens=512, top_k=50, temperature=0.1, **pipeline_kwargs
):
    """
    Load the LLM model and tokenizer, create the HuggingFace pipeline, and generate a RunnableSequence.

    Args:
        model: The pre-trained model loaded from Hugging Face.
        tokenizer: The tokenizer loaded from Hugging Face.
        max_new_tokens (int): The maximum number of new tokens to generate. Default is 200.
        top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering. Default is 50.
        temperature (float): The value used to module the next token probabilities. Default is 0.1.
        **pipeline_kwargs: Additional keyword arguments to pass to the HuggingFace pipeline.

    Returns:
        extract_answer (str): The generated response from the model.
    """
    # Create the HuggingFace pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        temperature=temperature,
        **pipeline_kwargs
    )

    # Wrap the pipeline with HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=pipe)

    # Create the RunnableSequence
    sequence = prompt_template | llm
    # Generate the query based on the user request
    return sequence


# Define a function to generate responses based on user input
def generate_query(user_query, sequence):
    inputs = {
        "user_query": user_query,
    }
    prompt = prompt_template.format(**inputs)

    response = sequence.invoke(inputs)
    extract_answer = response[len(prompt):]
    return extract_answer


# Example usage
# sequence = default_pipeline(model, tokenizer)
# user_query = "Do I need any special equipment to practice yoga?"
# output = generate_query(user_query, sequence)
