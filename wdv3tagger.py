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
    # CUDA/CPU check and reporting
    cuda_available = False
    try:
        import torch
        if torch.cuda.is_available():
            print("\n\033[92mCUDA detected! Using GPU acceleration\033[0m")
            cuda_available = True
        else:
            print("\n\033[93mCUDA not available - falling back to CPU\033[0m")
            cuda_available = False
    except ImportError:
        print("\n\033[91mPyTorch not installed - CPU only mode\033[0m")
        cuda_available = False

    sess_options = rt.SessionOptions()
    sess_options.log_severity_level = 2
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda_available else ['CPUExecutionProvider']
    model = rt.InferenceSession(model_path,
                              providers=providers,
                              sess_options=sess_options)
    print(f"Initialized with: {model.get_providers()}")
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

    def process_image_file(image_path, image_folder, output_to, remove_separator, final_tags):
        relative_path = os.path.relpath(image_path, image_folder)
        if output_to != "Metadata":
            caption_dir = os.path.join(output_path, os.path.dirname(relative_path))
            os.makedirs(caption_dir, exist_ok=True)
            caption_file_path = os.path.join(caption_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.txt")
        else:
            caption_file_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(image_path))[0]}.txt")
    
    
        final_tags_str = ", ".join(final_tags)
        if remove_separator:
            final_tags_str = final_tags_str.replace("_", " ")
    
        if output_to in ["Text File", "Both"]:
            try:
                with open(caption_file_path, 'w') as f:
                    f.write(final_tags_str)
                    print(f"Successfully processed {caption_file_path}")
            except Exception as e:
                print(f"Error processing {caption_file_path}: {str(e)}")
    
        if output_to in ["Metadata", "Both"]:
            update_metadata(image_path, final_tags)

    def update_metadata(image_path, final_tags):
        try:
            with ExifToolHelper() as et:
                existing = et.get_tags([image_path], ["IPTC:Keywords", "XMP:Subject"])[0]
                
                def normalize_tags(tags):
                    if isinstance(tags, list):
                        return [str(t).strip() for t in tags if t]
                    if isinstance(tags, str):
                        return [t.strip() for t in tags.split(",") if t.strip()]
                    return []
                
                iprc_list = normalize_tags(existing.get("IPTC:Keywords"))
                xmp_list = normalize_tags(existing.get("XMP:Subject"))
                
                all_tags = set(iprc_list + xmp_list).union(set(final_tags))
                
                et.set_tags(
                    [image_path],
                    tags={
                        "IPTC:Keywords": list(all_tags),
                        "XMP:Subject": list(all_tags)
                    },
                    params=["-P", "-overwrite_original"]
                )
            print(f"Successfully added tags to {image_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

    # Process all images
    def get_image_paths(img_folder: str, recurse: bool) -> iter:
        if recurse:
            for root, _, files in os.walk(img_folder):
                for file in files:
                    if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp')):
                        yield os.path.join(root, file)
        else:
            for file in os.listdir(img_folder):
                if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp')):
                    yield os.path.join(img_folder, file)

    try:
        for image_path in get_image_paths(image_folder, recursive):
            try:
                with Image.open(image_path) as image:
                    processed_image = prepare_image(image, target_size)
                    preds = model.run(None, {model.get_inputs()[0].name: processed_image})[0]

                final_tags = process_predictions_with_thresholds(
                    preds, tag_data, character_thresh, general_thresh,
                    hide_rating_tags, character_tags_first
                )

                process_image_file(image_path, image_folder, output_to, remove_separator, final_tags)
                processed_files.append(os.path.basename(image_path))
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
    except FileNotFoundError:
        error_message = "Error: The specified directory does not exist."
        print(error_message)  # Log the error to the console
        return error_message, ""
    
    return "Process completed.", "\n".join(processed_files)

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
    description="This tool tags all images in the specified directory and saves to .txt files inside 'captions' directory or embeds metadata directly into image files (supported formats: jpg (recommended), jpeg, png, bmp, gif, webp). Check 'Remove separator' to replace '_' with spaces in tags. Use Flag to generate a report which can be found in '.gradio' folder."
)

if __name__ == "__main__":
    iface.launch(inbrowser=True)
