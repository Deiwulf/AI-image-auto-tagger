# AI Image Auto Tagger

![ScreenCap.png](ScreenCap.png)


The successor of WD14 tagger and alternative to DeepDanbooru - now with metadata saving feature for nsfw-oriented gallery tagging (optimized for digiKam). Currently using wd-vit-tagger-v3 model (dataset as of 2024-02-28) by SmilingWolf which is newer than WD14 and DeepDanbooru. Using CUDA and ONNX library over Gradio WEBUI. Tested on Windows.

## Features
- **Supported models**: Latest WaifuDiffusion v3 tagger architecture featuring SmilingWolf/wd-vit-tagger-v3 model
- **Output**: Currently there are 3 output modes: .txt files, directly to image metadata, and both.
- **Supported Files**: jpg (recommended), jpeg, png, bmp, gif, webp
- **Easy User Interface**: By utilizing gradio for GUI, the usage of this script should be easy
- **Hide Rating Tags**: You can optionally choose whether to output the rating tags (Like "General", "Explicit", "Quetionable", etc) or not by checklist the "Hide Rating Tags" box
- **Character Tags First**: This feature makes the character name tag appear in front before other tags like general, copyright or rating. This feature is useful when training the text encoder with "keep n tokens"
- **Hide the Separator**: This function will remove the separator "_" of the tags in the output caption
- **User Preffered Threshold**: Using the gradio slider, the user can adjust the threshold of the tagger model

## How to run  
Python >3.10 and CUDA GPU is required to run this script. Download from [https://www.python.org/downloads](https://www.python.org/downloads/windows/)  
ExifTool >12.15 is required. Download from [https://exiftool.org](https://exiftool.org)

Steps to run:
1. Clone this repository `git clone https://github.com/Deiwulf/AI-image-auto-tagger.git` OR download as a zip and extract
2. Navigate to directory `cd AI-image-auto-tagger`
3. Set up a virtual environment `python -m venv venv` *
4. Activate the new venv: *
    - Windows: `venv\scripts\activate` 
5. Install the requirements `pip install -r requirements.txt`
6. Run the script `python wdv3tagger.py` OR use `start.bat` if on Windows (using venv)

\* Virtual environment is optional, but recommended to keep this isolated, you can skip to step 5 if you want to install and run on global environtment

## Disclaimer
This is still under development so please **do backups before running** and report issues.
