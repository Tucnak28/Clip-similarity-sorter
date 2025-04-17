# Semantic Image Sorter

Sort your images by visual and semantic similarity using OpenAI's CLIP model.

![preview](examples/example_output.png)

## Features
- Uses `sentence-transformers` implementation of CLIP
- Sorts images by similarity to a reference image
- Outputs renamed images into a separate folder

## Requirements
- Python 3.7+
- sentence-transformers
- Pillow
- torch

Install dependencies:
```bash
pip install -r requirements.txt
