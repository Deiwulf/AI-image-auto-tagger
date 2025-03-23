import os
import gradio as gr
import numpy as np
import pandas as pd
import onnxruntime as rt
from PIL import Image
import huggingface_hub
from exiftool import ExifToolHelper

# Define the path to save the text files / Lokasi untuk menyimpan output tags (.txt)
output_path = './captions/'

# Specific model repository from SmilingWolf's collection / Repository Default vit tagger v3
VIT_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-tagger-v3"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

# Download the model and labels
def download_model(model_repo):
    csv_path = huggingface_hub.hf_hub_download(model_repo, LABEL_FILENAME)
    model_path = huggingface_hub.hf_hub_download(model_repo, MODEL_FILENAME)
    return csv_path, model_path

# Load model and labels
def load_model(model_repo):
    csv_path, model_path = download_model(model_repo)
    tags_df = pd.read_csv(csv_path)
    tag_names = tags_df["name"].tolist()
    model = rt.InferenceSession(model_path)

    # Access the model target input size based on the model's first input details
    target_size = model.get_inputs()[0].shape[2] # Assuming the model input is square

    return model, tag_names, target_size

# Image preprocessing function / Memproses gambar
def prepare_image(image, target_size):
    canvas = Image.new("RGBA", image.size, (255, 255, 255))
    canvas.paste(image, mask=image.split()[3] if image.mode == 'RGBA' else None)
    image = canvas.convert("RGB")

    # Pad image to a square
    max_dim = max(image.size)
    pad_left = (max_dim - image.size[0]) // 2
    pad_top = (max_dim - image.size[1]) // 2
    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))

    # Resize
    padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

    # Convert to numpy array
    image_array = np.asarray(padded_image, dtype=np.float32)[..., [2, 1, 0]]
    
    return np.expand_dims(image_array, axis=0) # Add batch dimension

class LabelData:
    def __init__(self, names, rating, general, character):
        self.names = names
        self.rating = rating
        self.general = general
        self.character = character

def load_model_and_tags(model_repo):
    csv_path, model_path = download_model(model_repo)
    df = pd.read_csv(csv_path)
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )
    model = rt.InferenceSession(model_path)
    target_size = model.get_inputs()[0].shape[2]
    
    return model, tag_data, target_size

# Function to tag all images in a directory and save the captions / Fitur untuk tagging gambar dalam folder dan menyimpan caption dengan file .txt
def process_predictions_with_thresholds(preds, tag_data, character_thresh, general_thresh, hide_rating_tags, character_tags_first):
    # Extract prediction scores
    scores = preds.flatten()
    
    # Filter and sort character and general tags based on thresholds / Filter dan pengurutan tag berdasarkan ambang batas
    character_tags = [tag_data.names[i] for i in tag_data.character if scores[i] >= character_thresh]
    general_tags = [tag_data.names[i] for i in tag_data.general if scores[i] >= general_thresh]
    
    # Optionally filter rating tags
    rating_tags = [] if hide_rating_tags else [tag_data.names[i] for i in tag_data.rating]

    # Sort tags based on user preference / Mengurutkan tags berdasarkan keinginan pengguna
    final_tags = character_tags + general_tags if character_tags_first else general_tags + character_tags
    final_tags += rating_tags  # Add rating tags at the end if not hidden

    return final_tags

def tag_images(image_folder, recursive=False, character_tags_first=False, general_thresh=0.35, character_thresh=0.85, hide_rating_tags=True, remove_separator=False, output_to="Metadata"):
    if not image_folder:
        return "Error: Please provide a directory.", ""
    os.makedirs(output_path, exist_ok=True)
    model, tag_data, target_size = load_model_and_tags(VIT_MODEL_DSV3_REPO)

    # Process each image in the folder / Proses setiap gambar dalam folder
    processed_files = []

    if recursive:
        for root, _, files in os.walk(image_folder):
            for image_file in files:
                if image_file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp')):
                    image_path = os.path.join(root, image_file)
                    try:
                        with Image.open(image_path) as image:
                            processed_image = prepare_image(image, target_size)
                            preds = model.run(None, {model.get_inputs()[0].name: processed_image})[0]

                        final_tags = process_predictions_with_thresholds(preds, tag_data, character_thresh, general_thresh, hide_rating_tags, character_tags_first)

                        final_tags_str = ", ".join(final_tags)
                        if remove_separator:
                            final_tags_str = final_tags_str.replace("_", " ")

                        caption_file_path = os.path.join(output_path, f"{os.path.splitext(image_file)[0]}.txt")

                        if output_to in ["Text File", "Both"]:
                            with open(caption_file_path, 'w') as f:
                                f.write(final_tags_str)

                        if output_to in ["Metadata", "Both"]:
                            try:
                                with ExifToolHelper() as et:
                                    et.set_tags(
                                        [image_path],
                                        tags={
                                            "IPTC:Keywords": final_tags,
                                            "XMP:Subject": final_tags
                                        },
                                        params=["-P", "-overwrite_original"]
                                    )
                                print(f"Successfully added tags to {image_path}")
                            except Exception as e:
                                print(f"Error processing {image_path}: {str(e)}")

                        processed_files.append(image_file)
                    except Exception as e:
                        print(f"Error processing {image_path}: {str(e)}")
    else:
        for image_file in os.listdir(image_folder):
            if image_file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp')):
                image_path = os.path.join(image_folder, image_file)
                try:
                    with Image.open(image_path) as image:
                        processed_image = prepare_image(image, target_size)
                        preds = model.run(None, {model.get_inputs()[0].name: processed_image})[0]

                    final_tags = process_predictions_with_thresholds(preds, tag_data, character_thresh, general_thresh, hide_rating_tags, character_tags_first)

                    final_tags_str = ", ".join(final_tags)
                    if remove_separator:
                        final_tags_str = final_tags_str.replace("_", " ")

                    caption_file_path = os.path.join(output_path, f"{os.path.splitext(image_file)[0]}.txt")

                    if output_to in ["Text File", "Both"]:
                        with open(caption_file_path, 'w') as f:
                            f.write(final_tags_str)

                    if output_to in ["Metadata", "Both"]:
                        try:
                            with ExifToolHelper() as et:
                                et.set_tags(
                                    [image_path],
                                    tags={
                                        "IPTC:Keywords": final_tags,
                                        "XMP:Subject": final_tags
                                    },
                                    params=["-P", "-overwrite_original"]
                                )
                            print(f"Successfully added tags to {image_path}")
                        except Exception as e:
                            print(f"Error processing {image_path}: {str(e)}")

                    processed_files.append(image_file)
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")

    # Return both a completion message and a newline-separated list of processed files / Mengeluarkan pesan penyelesaian
    return "Process completed. Check caption files in the 'captions' directory.", "\n".join(processed_files)

iface = gr.Interface(
    fn=tag_images,
    inputs=[
        gr.Textbox(label="Enter the path to the image directory"),
        gr.Checkbox(label="Process subdirectories", value=False),
        gr.Checkbox(label="Character tags first"),
        gr.Slider(minimum=0, maximum=1, step=0.01, value=0.35, label="General tags threshold"),
        gr.Slider(minimum=0, maximum=1, step=0.01, value=0.85, label="Character tags threshold"),
        gr.Checkbox(label="Hide rating tags", value=True),
        gr.Checkbox(label="Remove separator", value=False),
        gr.Radio(choices=["Text File", "Both", "Metadata"], value="Metadata", label="Output to")
    ],
    outputs=[
        gr.Textbox(label="Status"),
        gr.Textbox(label="Processed Files")
    ],
    title="Image Captioning and Tagging with SmilingWolf/wd-vit-tagger-v3",
    description="This tool tags all images in the specified directory and saves to .txt files or embeds metadata directly into image files (supported formats: jpg (recommended), jpeg, png, bmp, gif, webp). Check 'Remove separator' to replace '_' with spaces in tags. Use Flag to generate a report which can be found in '.gradio' folder."
)

if __name__ == "__main__":
    iface.launch(inbrowser=True)
