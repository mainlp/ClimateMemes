from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import os
import torch
from tqdm import tqdm
import pandas as pd

def process_image(image_path):
    # Load and preprocess image as tensor
    return Image.open(image_path)

def generate_image_embeddings(model, processor, image):
    # Preprocess the image and text inputs
    inputs = processor.process(
        images=[image], 
        text='',  
        return_tensors="pt"
    )
    
    # Move inputs to the correct device and ensure batching
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    
    # Vision backbone outputs: image_features and cls_embed
    with torch.no_grad():
        image_features, cls_embed = model.model.vision_backbone(inputs['images'], inputs['image_masks'])
    
    num_image, num_patch = image_features.shape[1:3]
    image_features = image_features.view(1, num_image * num_patch, -1)

    return  image_features.mean(dim=1).squeeze()

def generate_text_embeddings(model, processor, ocr):
    inputs = processor.process(text=ocr)
    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    input_ids = inputs['input_ids']

    if input_ids is not None:
        input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)

    llm = model.model
    input_embedding = llm.transformer.wte(input_ids)[:, 3:-2, :]

    return input_embedding.mean(dim=1).squeeze()


if __name__ == "__main__":

    # Initialize model and processor
    cache_dir = "./.cache"
    device = "cuda:0"

    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map={"": device}
    )
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map={"": device}
    )

    # Example usage
    image_folder = "./images"

    excel_path = "./gold_label_with_frames.xlsx"
    df = pd.read_excel(excel_path)

    # Step 1: Load all images and extract embeddings
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_paths)} images in {image_folder}")

    # Create the folder if it doesn't exist
    embedding_folder = 'embeddings'
    os.makedirs(embedding_folder, exist_ok=True)

    # Initialize dictionary to hold all embeddings
    all_embeddings = {}

    for image_path in tqdm(image_paths, desc="Processing images"):
        # print(f"Processing {os.path.basename(image_path)}...")
        vision_input = process_image(image_path)

        image_name = os.path.basename(image_path) 
        matched_rows = df[df["Image"] == image_name]  
        ocr = matched_rows.iloc[0]["OCR Result"]

        print(f"Image: {image_name} -> OCR: {ocr}")

        image_embedding = generate_image_embeddings(model, processor, vision_input)
        text_embedding = generate_text_embeddings(model, processor, ocr)

        # Convert the tensor to CPU and store it in the dictionary with the image filename (without extension) as the key
        image_text_embedding = torch.cat([image_embedding, text_embedding], dim=-1)

        all_embeddings[image_name] = {
            "image": image_embedding.cpu(),
            "text": text_embedding.cpu(),
            "image_text": image_text_embedding.cpu()
        }
        print(image_embedding.shape, text_embedding.shape, image_text_embedding.shape)

    # Save the dictionary to a single .pth file
    embedding_file_path = os.path.join(embedding_folder, 'embedding_molmo_D.pth')
    torch.save(all_embeddings, embedding_file_path)
    print(f"Saved all embeddings to {embedding_file_path}")

