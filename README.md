# AI Image Auto Tagger

![ScreenCap.png](ScreenCap.png)


The successor of WD14 tagger and an alternative to DeepDanbooru - now with a metadata saving feature for NSFW-oriented gallery tagging (optimized for digiKam). Currently using wd-vit-tagger-v3 model by SmilingWolf which is newer than WD14 and latest DeepDanbooru. Using CUDA and ONNX library over Gradio WEBUI. Tested on Windows.

## Features
- **Output**: There are 2 output modes: embedding tags directly into images for galleries **OR** *.txt* files for AI training
- **Supported models**: Latest WaifuDiffusion v3 tagger architecture featuring SmilingWolf/wd-vit-tagger-v3 model
- **Supported files**: JPG/JPEG (recommended), PNG, WEBP, GIF, BMP(no metadata)
- **Easy user interface**: By utilizing gradio GUI, the usage of this script should be smooth
- **Process subdirectories**: recursively goes through all directories within a given one. In case of writing tags to *.txt*, it mirrors the folder structure 
- **User preferred threshold**: Using the gradio slider, the user can adjust the threshold of the tagger model
- **Hide rating tags**: You can choose whether to output the rating tags (Like "General", "Explicit", "Questionable", etc) or not
- **Character tags first**: This feature makes the character name tag appear in front of others
- **Remove separator**: This function will remove the standard separator "_" of the tags in the output caption
- **Overwrite existing metadata tags**: wipes clean any existing tags in metadata before writing new ones (XMP:Subject and IPTC:Keywords) 

## How to run  
Python >3.10 and CUDA GPU is required to run this script. Download from [https://www.python.org/downloads](https://www.python.org/downloads/windows/)  
ExifTool >12.15 is required. Download from [https://exiftool.org](https://exiftool.org)

Steps to run:
1. Clone this repository `git clone https://github.com/Deiwulf/AI-image-auto-tagger.git` OR download as a zip and extract
2. Navigate to the directory `cd AI-image-auto-tagger`
3. Set up a virtual environment `python -m venv venv` *
4. Activate the new venv: *
    - Windows: `venv\scripts\activate` 
5. Install the requirements `pip install -r requirements.txt`  
    - Optionally visit https://pytorch.org/ and install one fitting your system for a performance boost   
6. Run the script `python wdv3tagger.py` OR use `start.bat` on Windows (uses venv)

\* Virtual environment is optional, but recommended to keep this isolated. You can skip to step 5 if you want to install and run in a global environment

## Disclaimer
This has been thoroughly tested, but still do be savvy and back up before running.

## Was this free software useful to you?
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/N4N21UMI9H)
