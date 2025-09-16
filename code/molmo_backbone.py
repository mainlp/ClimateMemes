import argparse
import os
from openpyxl import Workbook
from tqdm import tqdm
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run Llava with n-shot setting.")
    parser.add_argument("--n_shot", type=int, default=0, help="Number of shots (default: 0).")
    parser.add_argument("--task", type=str, default="interpretation", help="Specify the task to perform. Options: 'interpretation', 'OCR', or 'detection'.")
    parser.add_argument("--meta", type=str, nargs='+', default=[], help="Data Augmentation.")
    return parser.parse_args()

def initialize_model_and_processor():
    """
    Initialize the model and processor with specified GPU IDs.

    Args:
        model_id (str): Model ID to load.

    Returns:
        tuple: Processor and model objects.
    """

    model_id = "Qwen/Qwen2-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir = "./.cache",
    torch_dtype="auto",
    device_map="auto"
)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer

def generate_rotated_options():
    # Step 1: Define the list of options
    options = [
    "This meme concerns scientific evidence regarding climate change",
    "This meme concerns the faithfulness of climate change representation",
    "This meme concerns human activities as a significant cause of climate change",
    "This meme concerns outcomes or consequences of climate change",
    "This meme concerns certain groups' responsibility to act on climate change",
    "This meme concerns the appropriateness of actions currently taken towards climate change",
    "This meme concerns the adequacy of current efforts in addressing climate change",
    "This meme concerns the impact of positive actions towards climate change"
    ]
    n = len(options)
    results = []
    # Step 2: Function to generate rotated sequences
    for i in range(n):
        rotated = options[i:] + options[:i]  # Rotate the options
        # Generate string with corresponding letters (A, B, ..., H)
        formatted = [
            f"{chr(65 + idx)}. {option}" for idx, option in enumerate(rotated)
        ]
        results.append("\n".join(formatted))  # Store as dictionary value
    return results

def get_similar_images(image_name, n_shot, similarity_df):
    """
    Retrieve the list of similar image paths for the given image name.

    Args:
        image_name (str): The current image name.
        n_shot (int): Number of similar images to retrieve.
        similarity_df (pd.DataFrame): The similarity DataFrame.
        image_folder (str): Folder containing images.

    Returns:
        list of str: Paths to the similar images.
    """
    train_image_folder = './images/train'
    if image_name in similarity_df['Image'].values:
        similar_images = similarity_df.loc[similarity_df['Image'] == image_name, 
                                           [f"Similar_{i+1}" for i in range(n_shot)]].values.flatten()
        # Convert similar image names to paths
        similar_image_paths = [os.path.join(train_image_folder, img) for img in similar_images if isinstance(img, str)]
        return similar_image_paths
    return []

def get_reference(image_name, gold_label_df, order):
    """
    Retrieve the reference answers for the given image from the gold label DataFrame,
    apply rotation based on order, and return formatted string for positions with "1".

    Args:
        image_name (str): The image name.
        gold_label_df (pd.DataFrame): The gold label DataFrame.
        order (int): The order for rotation (0-7).

    Returns:
        str: A formatted string indicating positions corresponding to "1".
    """
    # Extract the image name without URL prefix
    image_name_without_prefix = os.path.basename(image_name)
    
    # Check if the image name exists in the gold label DataFrame
    if image_name_without_prefix in gold_label_df['Meme Image URL'].str.split('/').str[-1].values:
        row = gold_label_df.loc[gold_label_df['Meme Image URL'].str.split('/').str[-1] == image_name_without_prefix]
        # Extract the answers for the 8 questions
        labels = row.iloc[0][["Question 5", "Question 6", "Question 7", "Question 8", 
                              "Question 9", "Question 10", "Question 11", "Question 12"]].tolist()
        
        # print(image_name, labels)
        # Rotate labels based on the given order
        rotated = labels[order:] + labels[:order]
        
        # Convert to binary: 0 for "c", 1 for anything else
        binary = [0 if label == "c" else 1 for label in rotated]
        
        # Map positions with "1" to corresponding letters (A-H)
        positions = [chr(65 + idx) for idx, value in enumerate(binary) if value == 1]
        
        # Return formatted string
        return f"<{', '.join(positions)}>"

    # Return empty placeholder if image is not found
    return "<>"

def get_ocr(image_name, gold_label_df):
    # Extract the image name without URL prefix
    image_name_without_prefix = os.path.basename(image_name)
    
    # Check if the image name exists in the gold label DataFrame
    if image_name_without_prefix in gold_label_df['Meme Image URL'].str.split('/').str[-1].values:
        row = gold_label_df.loc[gold_label_df['Meme Image URL'].str.split('/').str[-1] == image_name_without_prefix]
        ocr = row.iloc[0]["OCR Result"]
    return ocr

def get_human_interpretation(image_name, gold_label_df):
    # Extract the image name without URL prefix
    image_name_without_prefix = os.path.basename(image_name)
    
    # Check if the image name exists in the gold label DataFrame
    if image_name_without_prefix in gold_label_df['Meme Image URL'].str.split('/').str[-1].values:
        row = gold_label_df.loc[gold_label_df['Meme Image URL'].str.split('/').str[-1] == image_name_without_prefix]
        interpretation = row.iloc[0]["Question 4"]
    return interpretation

def get_gold_stance(image_name, gold_label_df):
    # Extract the image name without URL prefix
    image_name_without_prefix = os.path.basename(image_name)
    
    # Check if the image name exists in the gold label DataFrame
    if image_name_without_prefix in gold_label_df['Meme Image URL'].str.split('/').str[-1].values:
        row = gold_label_df.loc[gold_label_df['Meme Image URL'].str.split('/').str[-1] == image_name_without_prefix]
        interpretation = row.iloc[0]["Main Stance"]
    return interpretation

def get_model_interpretation(image_name, n_shot):
    model_interpretation_df = pd.read_excel(f'results/interpretation/molmo_{n_shot}_shot_interpretation.xlsx')
    # Extract the image name without URL prefix
    image_name_without_prefix = os.path.basename(image_name)
    
    # Check if the image name exists in the gold label DataFrame
    if image_name_without_prefix in model_interpretation_df['Image Name'].str.split('/').str[-1].values:
        row = model_interpretation_df.loc[model_interpretation_df['Image Name'].str.split('/').str[-1] == image_name_without_prefix]
        interpretation = row.iloc[0]["interpretation"]
    return interpretation

def generate_stance(model, tokenizer, n_shot, references, meta, ocrs=None, interpretations=None):
    """
    Generate answers using the model with n-shot examples, including correct answers as reference.

    Args:
        model: The loaded causal LM model.
        processor: The processor for the model.
        vision_inputs: List of preprocessed images (n-shot examples + current image).
        questions: List of questions to answer.
        n_shot: Number of shots (examples) to include in the context.
        references: List of reference answers for n-shot examples.

    Returns:
        List of answers generated for the questions.
    """
    answers = []

    prompt = ""

    conversation = [
    {"role": "system", "content": "Act as an advanced stance detection system specializing in climate change-related memes.\
    \n\
    1. Convinced: Accepts environmental risks, supports regulation of harmful activities, and reflects egalitarian and communitarian values.\
    \n\
    2. Skeptical: Downplays or denies environmental risks, opposes regulation, and prioritizes individual freedom and commerce.\
    \n\
    3. Neither: Does not align with convinced or skeptical stance and may present a neutral or unrelated stance.\
    '"}
]

    # Build the n-shot context with references
    if n_shot != 0:
        prompt += "Here are some examples:\n"
        for i in range(n_shot):
            if "ocr" in meta:
                if "human" in meta or "model" in meta:
                    prompt +=  f"Example meme {i+1}:\nThe following text is written inside the meme: {ocrs[i]}\nThis meme conveys: {interpretations[i]}\nStance: {references[i]}\n"
                else:
                    prompt +=  f"Example meme {i+1}:\nThe following text is written inside the meme: {ocrs[i]}\nStance: {references[i]}\n"
            else:
                if "human" in meta or "model" in meta:
                    prompt +=  f"Example meme {i+1}:\nThis meme conveys: {interpretations[i]}\nStance: {references[i]}\n"
                else:
                    prompt +=  f"Example meme {i+1}:\nStance: {references[i]}\n"

    if "ocr" in meta:
        if "human" in meta or "model" in meta:
            # print(meta)
            prompt +=  f"Can you evaluate the following meme:\nThe following text is written inside the meme: {ocrs[-1]}\nThis meme conveys: {interpretations[-1]}\nProvide the stance of this meme. ONLY ANSWER ONE WORD OF THE FOLLOWING: 'Convinced', 'Skeptical', or 'Neither'.\nStance:"
        else:
            prompt +=  f"Can you evaluate the following meme:\nThe following text is written inside the meme: {ocrs[-1]}\nProvide the stance of this meme. ONLY ANSWER ONE WORD OF THE FOLLOWING: 'Convinced', 'Skeptical', or 'Neither'.\nStance:"
    else:
        if "human" in meta or "model" in meta:
            prompt +=  f"Can you evaluate the following meme:\nThis meme conveys: {interpretations[-1]}\nProvide the stance of this meme. ONLY ANSWER ONE WORD OF THE FOLLOWING: 'Convinced', 'Skeptical', or 'Neither'.\nStance:"
        else:
            prompt += f"Can you evaluate the following meme:\nProvide the stance of this meme. ONLY ANSWER ONE WORD OF THE FOLLOWING: 'Convinced', 'Skeptical', or 'Neither'.\nStance:"

    conversation.append({
            "role": "user",
            "content": prompt
            })
    
    text = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True
)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )

    generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("A:", response)
    answers.append(response)

    return answers

def generate_frame(model, tokenizer, n_shot, references, options, meta, ocrs=None, interpretations=None):
    """
    Generate answers using the model with n-shot examples, including correct answers as reference.

    Args:
        model: The loaded causal LM model.
        processor: The processor for the model.
        vision_inputs: List of preprocessed images (n-shot examples + current image).
        questions: List of questions to answer.
        n_shot: Number of shots (examples) to include in the context.
        references: List of reference answers for n-shot examples.

    Returns:
        List of answers generated for the questions.
    """
    answers = []

    prompt = ""

    conversation = [
        {
        "role": "system",
        "content": f'You are an expert linguistic annotator.\n\
    We have collected hundreds of internet memes related to Climate Change.\n\
    Please read the question below carefully.\n\
    Your task is to select ALL media frames from the listed options below that are relevant to each meme.\n\
    Which ones of the following eight media frames apply to this meme (PLEASE SELECT ALL THAT APPLY)?\n{options}\n\
    '}]

    # Build the n-shot context with references
    if n_shot != 0:
        prompt += f"Here are some examples:\n"
        for i in range(n_shot):
            if "ocr" in meta:
                if "human" in meta or "model" in meta:
                    prompt += f"Example meme {i+1}:\nThe following text is written inside the meme: {ocrs[i]}\nThis meme conveys: {interpretations[i]}\nRelevant frames: {references[i]}\n"
                else:
                    prompt += f"Example meme {i+1}:\nThe following text is written inside the meme: {ocrs[i]}\nRelevant frames: {references[i]}\n"
            else:
                if "human" in meta or "model" in meta:
                    prompt += f"Example meme {i+1}:\nThis meme conveys: {interpretations[i]}\nRelevant frames: {references[i]}\n"
                else:
                    prompt += f"Example meme {i+1}:\nRelevant frames: {references[i]}\n"

    if "ocr" in meta:
        if "human" in meta or "model" in meta:
            # print(meta)
            prompt += f"Can you evaluate the following meme:\nThe following text is written inside the meme: {ocrs[-1]}\nThis meme conveys: {interpretations[-1]}\nGIVE ONLY THE OPTION LETTERS SEPARATED BY COMMA, nothing else.\nRelevant frames:"
        else:
            prompt += f"Can you evaluate the following meme:\nThe following text is written inside the meme: {ocrs[-1]}\nGIVE ONLY THE OPTION LETTERS SEPARATED BY COMMA, nothing else.\nRelevant frames:"
    else:
        if "human" in meta or "model" in meta:
            prompt +=  f"Can you evaluate the following meme:\nThis meme conveys: {interpretations[-1]}\nGIVE ONLY THE OPTION LETTERS SEPARATED BY COMMA, nothing else.\nRelevant frames:"
        else:
            prompt += f"Can you evaluate the following meme:\nGIVE ONLY THE OPTION LETTERS SEPARATED BY COMMA, nothing else.\nRelevant frames:"

    conversation.append({
            "role": "user",
            "content": prompt
            })
    
    text = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True
)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )

    generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("A:", response)
    answers.append(response)

    return answers

def save_answers_to_excel(image_folder, output_file, model, tokenizer, n_shot, similarity_df, gold_label_df, task, order=None, options=None, meta=None):
    """
    Save answers to an Excel file.

    Args:
        image_folder (str): Folder containing images.
        output_file (str): Path to save the Excel file.
        model: The loaded model.
        processor: The processor for the model.
        questions (list): The list of questions.
        n_shot (int): Number of shots to use.
        similarity_df (pd.DataFrame): DataFrame of image similarities.
        gold_label_df (pd.DataFrame): DataFrame of gold labels.
    """
    wb = Workbook()
    ws = wb.active
    ws.append(['Image Name'] + [task])

    for image_name in tqdm(os.listdir(image_folder), desc="Processing images"):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, image_name)

            # Get similar image paths
            similar_image_paths = get_similar_images(image_name, n_shot, similarity_df)
            all_image_paths = similar_image_paths + [image_path]  # Include current image
            # print(similar_image_paths, len(all_image_paths))

            # Get references (gold answers) for n_shot examples
            references = [] # frame references
            ocrs = []
            interpretations = []
            gold_interpretations = []
            gold_stances = []
            for similar_image_path in all_image_paths:
                similar_image_name = os.path.basename(similar_image_path)
                references.append(get_reference(similar_image_name, gold_label_df, order))
                ocrs.append(get_ocr(similar_image_name, gold_label_df))
                gold_interpretations.append(get_human_interpretation(similar_image_name, gold_label_df))
                gold_stances.append(get_gold_stance(similar_image_name, gold_label_df))
                if meta:
                    if "human" in meta:
                        interpretations.append(get_human_interpretation(similar_image_name, gold_label_df))
                    if "model" in meta:
                        interpretations.append(get_model_interpretation(similar_image_name, n_shot))

            if task == "frame":
                # Generate answers with n-shot references
                answers = generate_frame(model, tokenizer, n_shot, references, options, meta, ocrs, interpretations)
            
            elif task == "stance":
                # Generate answers with n-shot references
                answers = generate_stance(model, tokenizer, n_shot, gold_stances, meta, ocrs, interpretations)

            
            # Append answers to Excel
            ws.append([image_name] + answers)
            # print(f"{image_name} processed.")

    # Save results to Excel file
    wb.save(output_file)
    print(f"Results saved to {output_file}")


def main():
    """
    Main function to run the Llava-7B-D-0924 model for image question-answering with n-shot learning.
    """
    # Parse arguments
    args = parse_arguments()


    # Initialize model and processor with specified GPU ID
    # print(args.gpu_ids)
    model, tokenizer= initialize_model_and_processor()

    image_folder = './images/test'

    # Load similarity and gold label files
    similarity_df = pd.read_excel(f'results/ranking/image_text/image_text_similarity_molmo_D.xlsx')
    gold_label_df = pd.read_excel('./gold_label_with_frames.xlsx')

    if args.task == "frame":
        rotated_options = generate_rotated_options()

        # Run the process
        for order, options in enumerate(rotated_options):
            output_file = f'./results/{args.task}/Qwen2_{args.n_shot}_shot_{"_".join(args.meta)}_order{order}.xlsx'
            # print('order:', order)
            # print('options:', options)
            # print('output file:', output_file)
            save_answers_to_excel(
                image_folder=image_folder,
                output_file=output_file,
                model=model,
                tokenizer=tokenizer,
                n_shot=args.n_shot,
                similarity_df=similarity_df,
                gold_label_df=gold_label_df,
                order=order,
                options=options,
                task=args.task,
                meta=args.meta
            )

    elif args.task == "stance":
        output_file = f'./results/{args.task}/Qwen2_{args.n_shot}_shot_{"_".join(args.meta)}.xlsx'

        save_answers_to_excel(
                image_folder=image_folder,
                output_file=output_file,
                model=model,
                tokenizer=tokenizer,
                n_shot=args.n_shot,
                similarity_df=similarity_df,
                gold_label_df=gold_label_df,
                task=args.task,
                meta=args.meta,
            )

if __name__ == "__main__":
    main()
