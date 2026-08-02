# Image Captioning with Bahdanau Attention

An end-to-end Image Captioning system built from scratch in PyTorch using a ResNet50 encoder, Bahdanau Attention, and an LSTM decoder. The project includes beam search inference, attention visualization, and a deployed Streamlit web application for interactive image caption generation.

---

## Live Demo

**Web Application:** *https://imagecaptioningwithattention.streamlit.app/*

---

## Overview

Image Captioning is a multimodal AI task that combines Computer Vision and Natural Language Processing to generate descriptive captions for images.

This project implements an attention-based image captioning model from scratch in PyTorch. Instead of using a pretrained image captioning architecture, the model is built using a CNN encoder and an attention-based LSTM decoder.

Pipeline:

```
Image
   │
   ▼
ResNet50 Encoder
   │
   ▼
Spatial Feature Maps
   │
   ▼
Bahdanau Attention
   │
   ▼
LSTM Decoder
   │
   ▼
Generated Caption
```

---

## Features

- Generate captions for uploaded images
- Greedy Search decoding
- Beam Search decoding
- Attention heatmap visualization for each generated word
- Interactive Streamlit web interface
- Loads trained model directly from saved checkpoints
- Supports custom uploaded images without retraining

---

## Model Architecture

### Encoder

- ResNet50 CNN
- Extracts spatial image features
- Produces feature maps for the attention mechanism

### Attention

- Bahdanau (Additive) Attention
- Learns which image regions to focus on while generating each word

### Decoder

- Word Embedding Layer
- LSTMCell
- Fully Connected Output Layer
- Sequential caption generation

---

## Web Application

The deployed Streamlit application allows users to:

- Upload an image
- Generate captions using Beam Search
- View the original uploaded image
- Visualize attention heatmaps for every generated word

## Project Structure

```
Image-Captioning-With-Attention/
│
├── app.py                  # Streamlit application
├── inference.py            # Model loading and inference
├── best_model.pth          # Trained model weights
├── vocab.pth               # Vocabulary
├── requirements.txt
├── runtime.txt
├── README.md
│
├── data/                   # Dataset (excluded)
├── models/                 # Training files (excluded)
└── code.ipynb              # Development notebook
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/an1rudhrawat/Image-Captioning-With-Attention.git
cd Image-Captioning-With-Attention
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Pillow
- NLTK
- Scikit-learn

---

## Results

The deployed application supports:

- Greedy Search caption generation
- Beam Search caption generation
- Word-level attention visualization
- Interactive inference through a web interface

---

## Future Improvements

- Train on Flickr30k or MS COCO
- Transformer-based decoder
- Caption confidence scores
- Multiple decoding strategies
- Docker support
- REST API using FastAPI

