import json
import os
import random

import tqdm

from yogahub.data.llm.chat import generate_chat_completion


def process_questions(
    questions,
    default_instruction,
    train_test_ratio: float = 1.0,
    storage_folder: str = "./dataset",
):
    """
    Processes a list of questions by generating answers and saving them to train and test JSON files based on a specified ratio.

    Parameters:
    questions (list of str): List of questions to process.
    default_instruction (list of str): List of default instructions to randomly choose from.
    train_test_ratio (float): Ratio of the data to be used for training (e.g., 1 means 100% for training and 0% for testing).
    storage_folder (str): Folder to store the train and test JSON files.

    The function will generate a prompt for each question using a randomly selected instruction and a placeholder answer.
    It will then divide the questions into training and test sets based on the specified ratio and save them to "train.json" and "test.json" files respectively.

    Note: The actual answer generation logic should replace the placeholder "Generated answer" with a real function call.
    """

    if not os.path.exists(storage_folder):
        os.makedirs(storage_folder)

    length = int(len(questions) * train_test_ratio)
    n = 0

    for question in tqdm.tqdm(questions):
        instruction = random.choice(default_instruction)
        prompt = f"""
        {instruction}

        Question: {question}
        Answer (Providing an energetic and concise answer if possible, and please make response more varied):
        """

        # Generate chat completion (this part should be done outside this function and passed as 'answer')
        answer = generate_chat_completion(prompt)

        pair = {"instruction": instruction, "input": question, "output": answer}

        file_path = "train.json" if n <= length else "test.json"
        file_path = os.path.join(storage_folder, file_path)

        # Read existing data from the file
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        # Append the new pair to the existing data
        data.append(pair)

        # Write the updated data back to the file
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        n += 1


# Example usage:
# default_instruction = ["Now you are my enthusiastic yoga master. Please respond with energy and positivity as a yoga teacher to the following input.",
#                       "You are now my energetic yoga coach. Kindly guide me through the following query with enthusiasm and encouragement.",
#                       "Assume the role of my lively yoga instructor and provide a detailed, energetic response to the input.",
#                       "Imagine you are my vibrant yoga trainer. Answer the following question with enthusiasm and positivity.",
#                       "As my dynamic yoga teacher, please help me with the following information with energy and encouragement.",]
# with open("yogahub/data/llm/beginner_yoga_questions.txt", "r") as file:
#    questions = [line.strip() for line in file]
# process_questions(questions, default_instruction, 1, "dataset/")
