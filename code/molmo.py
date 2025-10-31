import os
import argparse
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import os
from openpyxl import Workbook
from tqdm import tqdm
import pandas as pd
import torch
from math import gcd

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run Molmo with n-shot setting.")
    # parser.add_argument("--t", type=int, default=7, help="Type of the model.")
    parser.add_argument("--task", type=str, default="frame", help="Specify the task to perform. Options: 'frame', 'stance', or 'interpretation'")
    parser.add_argument("--n_shot", type=int, default=0, help="Number of shots (default: 0).")
    parser.add_argument("--meta", type=str, nargs='+', default=[], help="Data Augmentation.")
    return parser.parse_args()


def initialize_model_and_processor():
    """
    Initialize the model and processor with a specified GPU ID.

    Args:
        gpu_id (int): The GPU ID to use.

    Returns:
        tuple: Processor and model objects.
    """
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    model_id = 'allenai/Molmo-7B-D-0924'

    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir="./.cache",
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir="./.cache",
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    return processor, model

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

def process_images(image_paths, target_size=(336, 336)):
    """
    Load and preprocess multiple images, resizing them to a fixed size.

    Args:
        image_paths (list of str): Paths to the images.
        target_size (tuple): The target size to resize images to, default is (256, 256).

    Returns:
        list of PIL.Image: Loaded and resized images.
    """
    images = []
    
    for image_path in image_paths:
        try:
            img = Image.open(image_path).convert("RGB")  # Convert to RGB
            
            img = img.resize(target_size, Image.LANCZOS)
            
            images.append(img)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
    
    return images

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

def get_frame_reference(image_name, gold_label_df, order):
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

def get_gold_frames(image_name, gold_label_df):
        # Extract the image name without URL prefix
    image_name_without_prefix = os.path.basename(image_name)
    
    # Check if the image name exists in the gold label DataFrame
    if image_name_without_prefix in gold_label_df['Meme Image URL'].str.split('/').str[-1].values:
        row = gold_label_df.loc[gold_label_df['Meme Image URL'].str.split('/').str[-1] == image_name_without_prefix]
        # Extract the answers for the 8 questions
        labels = row.iloc[0][["Question 5", "Question 6", "Question 7", "Question 8", 
                              "Question 9", "Question 10", "Question 11", "Question 12"]].tolist()
        
        # Convert to binary: 0 for "c", 1 for anything else
        binary = [0 if label == "c" else 1 for label in labels]

        options = [
    "REAL",
    "HOAX",
    "CAUSE",
    "IMPACT",
    "ALLOCATION",
    "PROPRIETY",
    "ADEQUACY",
    "PROSPECT"
    ]

        frames = ', '.join([options[i] for i in range(len(binary)) if binary[i] == 1])
        
        # Return formatted string
        return frames

    # Return empty placeholder if image is not found
    return ""

def get_pred_frames(image_name, pred_label_df):
        # Extract the image name without URL prefix
    image_name_without_prefix = os.path.basename(image_name)
    
    # Check if the image name exists in the gold label DataFrame
    if image_name_without_prefix in pred_label_df['Image Name'].values:
        row = pred_label_df.loc[pred_label_df['Image Name'] == image_name_without_prefix]
        # Extract the answers for the 8 questions
        labels = row.iloc[0]["frame"]
        options = {"A":"REAL",
    "B":"HOAX",
    "C":"CAUSE",
    "D":"IMPACT",
    "E":"ALLOCATION",
    "F":"PROPRIETY",
    "G":"ADEQUACY",
    "H":"PROSPECT"}

        frames = ', '.join([options.get(char, "") for char in labels if char in "ABCDEFGH"])
        print(frames)
        
        # Return formatted string
        return frames

    # Return empty placeholder if image is not found
    return ""

def get_model_interpretation(image_name, n_shot):
    model_interpretation_df = pd.read_excel(f'results/interpretation/molmo_{n_shot}_shot_interpretation.xlsx')

    # Extract the image name without URL prefix
    image_name_without_prefix = os.path.basename(image_name)
    
    print(image_name_without_prefix)
    # Check if the image name exists in the gold label DataFrame
    if image_name_without_prefix in model_interpretation_df['Image Name'].str.split('/').str[-1].values:
        row = model_interpretation_df.loc[model_interpretation_df['Image Name'].str.split('/').str[-1] == image_name_without_prefix]
        interpretation = row.iloc[0]["interpretation"]
    return interpretation
    
def generate_ocr(model, processor, vision_inputs):
    """
    Generate answers using the model with n-shot examples, including correct answers as reference.

    Args:
        model: The loaded causal LM model.
        processor: The processor for the model.
        vision_inputs: List of preprocessed images (n-shot examples + current image).

    Returns:
        List of answers generated for the questions.
    """
    answers = []

    # Create the prompt 
    context = f'Act as an advanced OCR (Optical Character Recognition) system.\
Extract all the text from the provided meme image.\
Focus on accurately identifying the text, including its font styles, formatting (e.g., bold, italic), and alignment (e.g., top-left, center).\
Do not interpret the text; simply provide the extracted content.'

    # Add the current image and question
    context += f'<im_start> Here is the meme image:\
<|image|> <im_end>\
\n\
Format your response as follows:\
\n\
Extracted Text: [Provide the extracted text here]\
\n\
Ensure that all text elements, including captions, embedded text, and small print, are included.'

    # Process all images together
    inputs = processor.process(
        images=vision_inputs,  # Preprocessed images (n-shot + current image)
        text=context
    )

    # Move inputs to the model's device and add batch dimension
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # Generate answer
    model.to(dtype=torch.bfloat16)
    inputs["images"] = inputs["images"].to(torch.bfloat16)

    model.eval()
    with torch.no_grad():
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=300, no_repeat_ngram_size=5, stop_strings=["<|endoftext|>"]),
            tokenizer=processor.tokenizer
        )
    # Extract generated tokens
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    answer = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    # print(f"Q:\n{context}\nA:\n{answer}")
    answers.append(answer)

    return answers

def generate_interpretation(model, processor, vision_inputs, n_shot, references):
    """
    Generate answers using the model with n-shot examples, including correct answers as reference.

    Args:
        model: The loaded causal LM model.
        processor: The processor for the model.
        vision_inputs: List of preprocessed images (n-shot examples + current image).
        n_shot: Number of shots (examples) to include in the context.
        references: List of reference answers for n-shot examples.

    Returns:
        List of answers generated for the questions.
    """
    answers = []
    # print("Run started...")  # Initial log

    # Create the prompt 
    context = f"You are an expert in meme analysis and text interpretation. Based on the provided meme image, analyze its meaning and how it contributes to the meme's humor, message, or context. Consider cultural references, tone, and any implied meanings."

    # Build the n-shot context with references
    if n_shot != 0:
        context += 'Here are some examples:\n'
        for i in range(n_shot):
            context += f"<im_start> Meme {i+1}: <|image|>\n"
            context += f"Meme Interpretation: {references[i]} <im_end>\n\n"

    # Add the current image and question
    context += f"<im_start> Here is the meme image:\
<|image|> <im_end>\
\n\
Format your response as follows:\
\n\
Meme Interpretation: [Provide your analysis here]\
\n\
Ensure your response is clear, concise, and focuses on the meme's potential intent or impact."

    # Process all images together
    inputs = processor.process(
        images=vision_inputs,  # Preprocessed images (n-shot + current image)
        text=context
    )

    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    model.eval()

    with torch.no_grad():

        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings=["<|endoftext|>"]),
            tokenizer=processor.tokenizer
        )

        generated_tokens = output[0, inputs['input_ids'].size(1):]

        answer = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        answers.append(answer)

    return answers


def generate_stance(model, processor, vision_inputs, n_shot, references, meta, ocrs=None, interpretations=None):

    answers = []
    # print("Run started...")  # Initial log

    context = f'Act as an advanced stance detection system specializing in climate change-related memes.\
    Your task is to classify the provided meme into one of the following stances based on its message, tone, and imagery:\
    \n\
    1. Convinced: Accepts environmental risks, supports regulation of harmful activities, and reflects egalitarian and communitarian values.\
    \n\
    2. Skeptical: Downplays or denies environmental risks, opposes regulation, and prioritizes individual freedom and commerce.\
    \n\
    3. Neither: Does not align with convinced or skeptical stance and may present a neutral or unrelated stance.\
    '

    # Build the n-shot context with references
    if n_shot != 0:
        context += 'Here are some examples:\n'
        for i in range(n_shot):
            context += f"<im_start> Meme {i+1}: <|image|>\n"

            if "ocr" in meta:
                context += f"The following text is written inside the meme: {ocrs[i]}\n"

            if "human" in meta:
                context += f"This meme conveys: {interpretations[i]}\n"

            if "model" in meta:
                context += f"This meme conveys: {interpretations[i]}\n"
            
            context += f"Stance: {references[i]} <im_end>\n\n"

    # Add the current image and question
    context += f"<im_start> Can you evaluate the following meme:\n\
Meme: <|image|>\n"
    
    if "ocr" in meta:
        context += f" {ocrs[-1]}\n"

    if "human" in meta:
        context += f"This meme conveys: {interpretations[-1]}\n"

    if "model" in meta:
        context += f"This meme conveys: {interpretations[-1]}\n"

    context += f"Provide the stance of this meme. ONLY ANSWER ONE WORD OF THE FOLLOWING: 'Convinced', 'Skeptical', or 'Neither'.\
    \n\
    Stance:"

    # Process all images together
    inputs = processor.process(
        images=vision_inputs,  # Preprocessed images (n-shot + current image)
        text=context
    )

    # Move inputs to the model's device and add batch dimension
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # Generate answer
    model.to(dtype=torch.bfloat16)
    inputs["images"] = inputs["images"].to(torch.bfloat16)

    model.eval()
    with torch.no_grad():
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=20, stop_strings=["<|endoftext|>"]),
            tokenizer=processor.tokenizer
        )
    # Extract generated tokens
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    answer = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Q:\n{context}\nA:\n{answer}")
    answers.append(answer)

    return answers

def generate_frame(model, processor, vision_inputs, n_shot, references, options, meta, ocrs=None, interpretations=None):
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

    # Create the prompt 
    context = f'You are an expert linguistic annotator.\n\
We have collected hundreds of internet memes related to Climate Change.\n\
Please read the question below carefully.\n\
Your task is to select ALL media frames from the listed options below that are relevant to each meme.\n\n\
Which ones of the following eight media frames apply to this meme (PLEASE SELECT ALL THAT APPLY)?\n{options}\n\n\
    '

    # Build the n-shot context with references
    if n_shot != 0:
        context += 'Here are some examples:\n'
        for i in range(n_shot):
            context += f"<im_start> Meme {i+1}: <|image|>\n"

            if "ocr" in meta:
                context += f"The following text is written inside the meme: {ocrs[i]}\n"

            if "human" in meta:
                context += f"This meme conveys: {interpretations[i]}\n"

            if "model" in meta:
                context += f"This meme conveys: {interpretations[i]}\n"
            
            context += f"Relevant frames: {references[i]} <im_end>\n\n"

    # Add the current image and question
    context += f"<im_start> Can you evaluate the following meme:\n\
Meme: <|image|>\n"
    
    if "ocr" in meta:
        context += f" {ocrs[-1]}\n"

    if "human" in meta:
        context += f"This meme conveys: {interpretations[-1]}\n"

    if "model" in meta:
        context += f"This meme conveys: {interpretations[-1]}\n"
    
    context += f"GIVE ONLY THE OPTION LETTERS SEPARATED BY COMMA, nothing else.\nRelevant frames:"

    # Process all images together
    inputs = processor.process(
        images=vision_inputs,  # Preprocessed images (n-shot + current image)
        text=context
    )

    # Move inputs to the model's device and add batch dimension
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # Generate answer
    model.to(dtype=torch.bfloat16)
    inputs["images"] = inputs["images"].to(torch.bfloat16)

    model.eval()
    with torch.no_grad():
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=20, stop_strings=["<|endoftext|>"]),
            tokenizer=processor.tokenizer
        )
    # Extract generated tokens
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    answer = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    # print(f"Q:\n{context}\nA:\n{answer}")
    answers.append(answer)

    return answers

def generate_stance_augment(model, processor, vision_inputs, n_shot, gold_stances, references, meta, ocrs=None, interpretations=None):

    answers = []
    # print("Run started...")  # Initial log

    context = f'Act as an advanced stance detection system specializing in climate change-related memes.\
    Your task is to classify the provided meme into one of the following stances based on its message, tone, and imagery:\
    \n\
    1. Convinced: Accepts environmental risks, supports regulation of harmful activities, and reflects egalitarian and communitarian values.\
    \n\
    2. Skeptical: Downplays or denies environmental risks, opposes regulation, and prioritizes individual freedom and commerce.\
    \n\
    3. Neither: Does not align with convinced or skeptical stance and may present a neutral or unrelated stance.\
    '

    # Build the n-shot context with references
    if n_shot != 0:
        context += 'Here are some examples:\n'
        for i in range(n_shot):
            context += f"<im_start> Meme {i+1}: <|image|>\n"

            if "ocr" in meta:
                context += f"The following text is written inside the meme: {ocrs[i]}\n"

            if "human" in meta:
                context += f"This meme conveys: {interpretations[i]}\n"

            if "model" in meta:
                context += f"This meme conveys: {interpretations[i]}\n"
            
            context += f"This meme uses Frames: {references[i]}\n"
            
            context += f"Stance: {gold_stances[i]} <im_end>\n\n"

    # Add the current image and question
    context += f"<im_start> Can you evaluate the following meme:\n\
Meme: <|image|>\n"
    
    if "ocr" in meta:
        context += f" {ocrs[-1]}\n"

    if "human" in meta:
        context += f"This meme conveys: {interpretations[-1]}\n"

    if "model" in meta:
        context += f"This meme conveys: {interpretations[-1]}\n"
    
    context += f"This meme uses Frames: {references[-1]}\n"

    context += f"Provide the stance of this meme. ONLY ANSWER ONE WORD OF THE FOLLOWING: 'Convinced', 'Skeptical', or 'Neither'.\
    \n\
    Stance:"

    # Process all images together
    inputs = processor.process(
        images=vision_inputs,  # Preprocessed images (n-shot + current image)
        text=context
    )

    # Move inputs to the model's device and add batch dimension
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # Generate answer
    model.to(dtype=torch.bfloat16)
    inputs["images"] = inputs["images"].to(torch.bfloat16)

    model.eval()
    with torch.no_grad():
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=20, stop_strings=["<|endoftext|>"]),
            tokenizer=processor.tokenizer
        )
    # Extract generated tokens
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    answer = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Q:\n{context}\nA:\n{answer}")
    answers.append(answer)

    return answers

def save_answers_to_excel(image_folder, output_file, model, processor, n_shot, similarity_df, gold_label_df, task, cot=True, order=None, options=None, meta=None):
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

    pred_label_df = pd.read_excel('results/frame/Molmo_4_shot_ocr_order0.xlsx')

    for image_name in tqdm(os.listdir(image_folder), desc="Processing images"):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, image_name)

            # Get similar image paths
            similar_image_paths = get_similar_images(image_name, n_shot, similarity_df)
            all_image_paths = similar_image_paths + [image_path]  # Include current image

            # Load all images
            vision_inputs = process_images(all_image_paths)
            # print(f"Images loaded for {image_name}: {all_image_paths}")

            # Get references (gold answers) for n_shot examples
            references = [] 
            ocrs = []
            interpretations = []
            gold_interpretations = []
            gold_stances = []
            gold_frames = []
            pred_frames = []
            for similar_image_path in all_image_paths:
                similar_image_name = os.path.basename(similar_image_path)
                references.append(get_frame_reference(similar_image_name, gold_label_df, order))
                ocrs.append(get_ocr(similar_image_name, gold_label_df))
                gold_interpretations.append(get_human_interpretation(similar_image_name, gold_label_df))
                gold_stances.append(get_gold_stance(similar_image_name, gold_label_df))
                gold_frames.append(get_gold_frames(similar_image_name, gold_label_df))
                pred_frames.append(get_pred_frames(similar_image_name, pred_label_df))
                print(similar_image_path, get_pred_frames(similar_image_name, pred_label_df))
                if meta:
                    if "human" in meta:
                        interpretations.append(get_human_interpretation(similar_image_name, gold_label_df))
                    if "model" in meta:
                        interpretations.append(get_model_interpretation(similar_image_name, n_shot))
            
            if task == "interpretation":
                answers = generate_interpretation(model, processor, vision_inputs, n_shot, gold_interpretations)

            elif task == "ocr":
                answers = generate_ocr(model, processor, vision_inputs)

            elif task == "stance":
                answers = generate_stance(model, processor, vision_inputs, n_shot, gold_stances, meta, ocrs, interpretations)

            elif task == "frame":
            # Generate answers with n-shot references
                answers = generate_frame(model, processor, vision_inputs, n_shot, references, options, meta, ocrs, interpretations)

            elif task == "augment":
                if cot:
                    answers = generate_stance_augment(model, processor, vision_inputs, n_shot, gold_stances, pred_frames, meta, ocrs, interpretations)
                else:
                    answers = generate_stance_augment(model, processor, vision_inputs, n_shot, gold_stances, gold_frames, meta, ocrs, interpretations)
                break
            
            # Append answers to Excel
            ws.append([image_name] + answers)
            # print(f"{image_name} processed.")

    # Save results to Excel file
    wb.save(output_file)
    print(f"Results saved to {output_file}")

def main():
    """
    Main function to run the Molmo-7B-D-0924 model for image question-answering with n-shot learning.
    """
    # Parse arguments
    args = parse_arguments()

    processor, model = initialize_model_and_processor()

    # Load similarity and gold label files
    similarity_df = pd.read_excel(f'results/ranking/image_text/image_text_similarity_molmo_D.xlsx')
    gold_label_df = pd.read_excel('./gold_label_with_frames.xlsx')

    image_folder = './images/test'

    if args.task == "ocr":
        output_file = f'./results/ocr/molmo.xlsx'

        save_answers_to_excel(
                image_folder=image_folder,
                output_file=output_file,
                model=model,
                processor=processor,
                n_shot=args.n_shot,
                similarity_df=similarity_df,
                gold_label_df=gold_label_df,
                task=args.task
            )
    
    elif args.task == "interpretation":
        output_file = f'./results/interpretation/Molmo_{args.n_shot}_shot_interpretation.xlsx'

        save_answers_to_excel(
                image_folder=image_folder,
                output_file=output_file,
                model=model,
                processor=processor,
                n_shot=args.n_shot,
                similarity_df=similarity_df,
                gold_label_df=gold_label_df,
                task=args.task
            )

    elif args.task == "frame":
        rotated_options = generate_rotated_options()

        # Run the process
        for order, options in enumerate(rotated_options):
            output_file = f'./results/{args.task}/Molmo_{args.n_shot}_shot_{"_".join(args.meta)}_order{order}.xlsx'
            save_answers_to_excel(
                image_folder=image_folder,
                output_file=output_file,
                model=model,
                processor=processor,
                n_shot=args.n_shot,
                similarity_df=similarity_df,
                gold_label_df=gold_label_df,
                order=order,
                options=options,
                task=args.task,
                meta=args.meta
            )

    if args.task == "stance":
        output_file = f'./results/{args.task}/Molmo_{args.n_shot}_shot_{"_".join(args.meta)}.xlsx'

        save_answers_to_excel(
                image_folder=image_folder,
                output_file=output_file,
                model=model,
                processor=processor,
                n_shot=args.n_shot,
                similarity_df=similarity_df,
                gold_label_df=gold_label_df,
                task=args.task,
                meta=args.meta
            )
    
    if args.task == "augment":
        output_file = f'./results/{args.task}_cot/Molmo_{args.n_shot}_shot_{"_".join(args.meta)}.xlsx'

        save_answers_to_excel(
                image_folder=image_folder,
                output_file=output_file,
                model=model,
                processor=processor,
                n_shot=args.n_shot,
                similarity_df=similarity_df,
                gold_label_df=gold_label_df,
                task=args.task,
                meta=args.meta
            )

if __name__ == "__main__":
    main()
