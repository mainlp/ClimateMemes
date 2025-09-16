from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from PIL import Image
import os
import torch
from tqdm import tqdm

def process_image(image_path):
    """
    Load and preprocess an image for model input.
    Convert the image to RGB format.
    """
    return Image.open(image_path).convert("RGB")

def generate_image_embeddings(model, processor, image):
    # Create a conversational prompt for the image
    conversation = [
        {
            "role": "user",
            "content": [{"type": "text", "text": ''}, {"type": "image"}],
        }
    ]
    # Apply the processor's chat template to prepare the input
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)
    # Preprocess the image and text inputs
    inputs = processor(
        images=image, 
        text=prompt,  
        return_tensors="pt"
    ).to("cuda:2")

    # Perform a forward pass through the model to get embeddings
    with torch.no_grad():
        image_features = model.get_image_features(inputs["pixel_values"],
                                          inputs['image_sizes'],
                                          model.config.vision_feature_layer,
                                          model.config.vision_feature_select_strategy)
        # Use the last hidden state and perform mean pooling
        embedding = image_features[0].mean(dim=(0, 1))
        print(embedding.shape)
    return embedding

if __name__ == "__main__":

    # Initialize the LLaVA model and processor
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    cache_dir = ""

    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir,
        torch_dtype=torch.float16, 
        device_map={"": "cuda:5"}
    )
    processor = LlavaNextProcessor.from_pretrained(model_id)

    # Example usage
    image_folder = "./images"
    embedding_folder = 'embedding/image'

    # Step 1: Load all images and extract embeddings
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_paths)} images in {image_folder}")

    # Create the folder if it doesn't exist
    embedding_folder = 'embedding/image'
    os.makedirs(embedding_folder, exist_ok=True)

    # Initialize dictionary to hold all embeddings
    all_embeddings = {}

    for image_path in tqdm(image_paths, desc="Processing images"):
        print(f"Processing {os.path.basename(image_path)}...")
        vision_input = process_image(image_path)
        embedding = generate_image_embeddings(model, processor, vision_input)

        # Convert the tensor to CPU and store it in the dictionary with the image filename (without extension) as the key
        image_name = os.path.basename(image_path)
        all_embeddings[image_name] = embedding.cpu()

    # Save the dictionary to a single .pth file
    embedding_file_path = os.path.join(embedding_folder, 'image_embedding_llava.pth')
    torch.save(all_embeddings, embedding_file_path)
    print(f"Saved all embeddings to {embedding_file_path}")