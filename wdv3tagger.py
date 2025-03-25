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

# File extension support and MIME type mapping
type_map = {
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'png': 'image/png',
    'gif': 'image/gif',
    'bmp': 'image/bmp',
    'webp': 'image/webp'
}

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

# Gather all tags as per user settings
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

# Check whether image extensions are set correctly and filter out unsupported ones
def validate_file_format(file_path: str, output_to) -> tuple:
   
    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    
    if output_to == "Metadata" and ext == "bmp":
        msg = "BMP metadata not supported"
        return (False, msg, file_path)
    
    try:
        with ExifToolHelper() as et:
            metadata = et.get_tags([file_path], 'File:MIMEType')[0]
            actual_mime = metadata.get('File:MIMEType', '')
    except Exception as e:
        msg = f"Error reading metadata: {str(e)}"
        return (False, msg, file_path)
    
    # Check if the actual MIME type is supported
    if actual_mime not in type_map.values():
        msg = f"Unsupported format: {actual_mime}"
        return (False, msg, file_path)

    # Attempt to correct the extension
    correct_ext = next((k for k, v in type_map.items() if v in actual_mime), None)
    if ext != correct_ext:
        if correct_ext:
            new_file_path = os.path.splitext(file_path)[0] + '.' + correct_ext
            if not os.path.exists(new_file_path):
                try:
                    os.rename(file_path, new_file_path)
                    print(f"Auto-corrected extension: Renamed '{os.path.basename(file_path)}' to '{os.path.basename(new_file_path)}'")
                    return (True, None, new_file_path)
                except Exception as e:
                    msg = f"Error correcting extension: {str(e)}"
                    return (False, msg, file_path)
            else:
                msg = f"Format mismatch: {os.path.basename(file_path)} has .{ext} extension but actual format is {actual_mime} and renaming would overwrite an existing file."
                return (False, msg, file_path)
        else:
            msg = f"Format mismatch: {os.path.basename(file_path)} has .{ext} extension but actual format is {actual_mime}. Could not determine correct extension."
            return (False, msg, file_path)

    return (True, None, file_path)

# MAIN
def tag_images(image_folder, recursive=False, general_thresh=0.35, character_thresh=0.85, hide_rating_tags=True, character_tags_first=False, remove_separator=False, overwrite_tags=False, output_to="Metadata"):
    if not image_folder:
        return "Error: Please provide a directory.", "", ""
    os.makedirs(output_path, exist_ok=True)
    model, tag_data, target_size = load_model_and_tags(VIT_MODEL_DSV3_REPO)

    # Process each image in the folder / Proses setiap gambar dalam folder
    processed_files = []
    skipped_files = []

    def normalize_tags(tags):
        if isinstance(tags, list):
            return [str(t).strip() for t in tags if t]
        if isinstance(tags, str):
            return [t.strip() for t in tags.split(",") if t.strip()]
        return []

    def update_metadata(image_path, final_tags, overwrite_tags):
        try:
            with ExifToolHelper() as et:
                existing = et.get_tags([image_path], ["IPTC:Keywords", "XMP:Subject"])[0]
                
                iptc_list = normalize_tags(existing.get("IPTC:Keywords"))
                xmp_list = normalize_tags(existing.get("XMP:Subject"))

                if not overwrite_tags:
                    combined_tags = final_tags + iptc_list + xmp_list
                else:
                    combined_tags = final_tags

                # Remove duplicates while preserving order
                all_tags = list(dict.fromkeys(combined_tags))

                et.set_tags(
                    [image_path],
                    tags={
                        "IPTC:Keywords": all_tags,
                        "XMP:Subject": all_tags
                    },
                    params=["-P", "-overwrite_original"]
                )
            print(f"Successfully added tags to {image_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

    def process_image_file(image_path, image_folder, output_to, remove_separator, final_tags, overwrite_tags):
        relative_path = os.path.relpath(image_path, image_folder)

        if output_to == "Metadata":
            if remove_separator:
                final_tags = [tag.replace("_", " ") for tag in final_tags]
            update_metadata(image_path, final_tags, overwrite_tags)
        
        if output_to == "Text File":
            # Determine the caption file path
            caption_dir = os.path.join(output_path, os.path.dirname(relative_path))
            os.makedirs(caption_dir, exist_ok=True)
            caption_file_path = os.path.join(caption_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.txt")
    
            final_tags_str = ", ".join(final_tags)
            if remove_separator:
                final_tags_str = final_tags_str.replace("_", " ")
    
            try:
                with open(caption_file_path, 'w') as f:
                    f.write(final_tags_str)
                    print(f"Successfully processed {caption_file_path}")
            except Exception as e:
                print(f"Error processing {caption_file_path}: {str(e)}")

    # Yield image paths with validated file formats
    def get_image_paths(img_folder: str, recurse: bool) -> iter:
        
        if recurse:
            for root, _, files in os.walk(img_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.isfile(file_path):
                        valid, msg, file_path = validate_file_format(file_path, output_to)
                        if not valid:
                            print(f"Skipping {file_path}: {msg}")
                            skipped_files.append(os.path.basename(file_path))
                            continue
                        yield file_path
        else:
            for file in os.listdir(img_folder):
                file_path = os.path.join(img_folder, file)
                if os.path.isfile(file_path):   
                    valid, msg, file_path = validate_file_format(file_path, output_to)
                    if not valid:
                        print(f"Skipping {file_path}: {msg}")
                        skipped_files.append(os.path.basename(file_path))
                        continue
                    yield file_path

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

                process_image_file(image_path, image_folder, output_to, remove_separator, final_tags, overwrite_tags)

                if os.path.basename(image_path) not in skipped_files:
                    processed_files.append(os.path.basename(image_path))
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                skipped_files.append(os.path.basename(image_path))
    except FileNotFoundError:
        error_message = f"Error: The specified directory does not exist."
        print(error_message)
        return error_message, "", ""
    
    
    status_message = f"DONE -- Processed files: {len(processed_files)} -- Skipped files: {len(skipped_files)} -- See console for more details"
    print("\033[92mDONE\033[0m")
    return status_message, "\n".join(processed_files), "\n".join(skipped_files)

iface = gr.Interface(
    fn=tag_images,
    inputs=[
        gr.Textbox(label="Enter the path to the image directory"),
        gr.Checkbox(label="Process subdirectories", value=False),
        gr.Slider(minimum=0, maximum=1, step=0.01, value=0.35, label="General tags threshold"),
        gr.Slider(minimum=0, maximum=1, step=0.01, value=0.85, label="Character tags threshold"),
        gr.Checkbox(label="Hide rating tags", value=True),
        gr.Checkbox(label="Character tags first"),
        gr.Checkbox(label="Remove separator", value=False),
        gr.Checkbox(label="Overwrite existing metadata tags", value=False),
        gr.Radio(choices=["Text File", "Metadata"], value="Metadata", label="Output to")
        
    ],
    outputs=[
        gr.Textbox(label="Status"),
        gr.Textbox(label="Processed Files"),
        gr.Textbox(label="Skipped Files")
    ],
    title="Image Captioning and Tagging with SmilingWolf/wd-vit-tagger-v3",
    description="This tool tags all images in the specified directory and saves to .txt files inside 'captions' directory or embeds metadata directly into image files (supported formats: JPG/JPEG (recommended), PNG, GIF, WEBP, BMP(not metadata)). Check 'Remove separator' to replace '_' with spaces in tags. Use Flag to generate a report which can be found in '.gradio' folder."
)

if __name__ == "__main__":
    iface.launch(inbrowser=True)
