import os
# import pandas as pd
import numpy as np
from PIL import Image
# from sklearn.model_selection import train_test_split
# from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
# from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import torch.nn.functional as F
import math
from nltk.corpus import stopwords


device = ("cuda" if torch.cuda.is_available() else "cpu")
device

class Vocabulary:
    def __init__(self, min_freq = 5):
        
        # threshold
        self.min_freq = min_freq

        # special characters / tokens
        self.pad_token = "<pad>"
        self.start_token = "<start>"
        self.end_token = "<end>"
        self.unk_token = "<unk>"

        # word -> indx mapping
        self.word2idx = {
            self.pad_token : 0,
            self.start_token : 1,
            self.end_token : 2,
            self.unk_token : 3
        }

        # indx -> word mapping
        self.idx2word = {
            0 : self.pad_token,
            1 : self.start_token,
            2 : self.end_token,
            3 : self.unk_token
        }

        # Counter = counts words frequency
        self.word_freq = {}

    def build_vocabulary(self, captions: list[str]) -> None:
        """
        Building vocabulary from training captions
        """
        for sentence in captions:
            for word in sentence.lower().split():
                if word in self.word_freq:
                    self.word_freq[word] += 1
                else:
                    self.word_freq[word] = 1

        # Words which pass the threshold
        for word in self.word_freq:
            if self.word_freq[word] >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def numericalize(self, caption: str) -> list[int]:
        """
        Convert a string into a list of token
        """
        tokens = [self.word2idx[self.start_token]]
        for word in caption.lower().split():
            if word in self.word2idx:
                tokens.append(self.word2idx[word])
            else:
                tokens.append(self.word2idx[self.unk_token])
        tokens.append(self.word2idx[self.end_token])
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """ 
        Numerical to string value using the tokens
        """
        words = []
        for val in tokens:
            words.append(
                self.idx2word.get(val, self.unk_token)
            )
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet50(
            weights = models.ResNet50_Weights.DEFAULT
        )

        # Removing last two layers as we only need to extract features
        modules = list(resnet.children())[: -2]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        features = self.resnet(images)
        features = features.flatten(start_dim = 2)
        features = features.permute(0, 2, 1)
        return features


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()

        self.encoder_att = nn.Linear(
            encoder_dim,
            attention_dim
        )

        self.decoder_att = nn.Linear(
            decoder_dim,
            attention_dim
        )

        self.full_att = nn.Linear(
            attention_dim,
            1
        )

        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, encoder_features, hidden_state):
        encoder_projection = self.encoder_att(encoder_features)
        decoder_projection = self.decoder_att(hidden_state)
        decoder_projection = decoder_projection.unsqueeze(1)

        # Raw scores for every area
        score = self.full_att(
            torch.tanh(
                encoder_projection + decoder_projection
            )
        ).squeeze(2)

        attention_weights = self.softmax(score)
        context_vector = (
            attention_weights.unsqueeze(2) * encoder_features
        ).sum(dim = 1)
        return context_vector, attention_weights


class DecoderWithAttention(nn.Module):
    def __init__(
        self,
        attention,
        embed_size,
        hidden_size,
        vocab_size,
        encoder_dim = 2048,
        dropout = 0.5
    ):
        super().__init__()
        self.attention = attention

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size
        )
        self.lstm = nn.LSTMCell(
            embed_size + encoder_dim,
            hidden_size
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )
    def forward(self, encoder_features, captions):
        batch_size = encoder_features.size(0)
        max_len = captions.size(1)

        # Convert token id to word embeddings
        embeddings = self.embedding(captions)

        # hidden state for short term
        h = torch.zeros(
            batch_size,
            self.lstm.hidden_size,
            device = encoder_features.device
        )

        # cell state for long term
        c = torch.zeros_like(h)
        predictions = []
        for i in range(max_len - 1):
            context, alpha = self.attention(
                encoder_features,
                h
            )

            # extract the embedding of the current input word
            word_embedding = embeddings[:, i, :]

            # concat word embedding and attention image features
            lstm_input = torch.cat(
                [word_embedding, context],
                dim = 1
            )

            # update hidden and cell state
            h, c = self.lstm(
                lstm_input,
                (h, c)
            )
            prediction = self.fc(
                self.dropout(h)
            )
            predictions.append(prediction)

        # convert list of predictions into a single tensor
        predictions = torch.stack(
            predictions,
            dim = 1
        )

        return predictions

    # --------------------------------------------------------------
    # --------------------------------------------------------------
    # --------------------------------------------------------------
    # --------------------------------------------------------------
    # --------------------------------------------------------------

        # helper function for a single decoding step
    def decode_step(
        self,
        encoder_features,
        current_word,
        hidden,
        cell
    ):
        embedding = self.embedding(current_word)
        
        context, alpha = self.attention( # alpha is attention weights
            encoder_features,
            hidden
        )

        # concat embedding and context
        lstm_input = torch.cat(
            [embedding.squeeze(1), context], dim = 1
        )

        # LSTM cell
        hidden, cell = self.lstm(
            lstm_input,
            (hidden, cell)
        )

        # Prediction
        predictions = self.fc(
            self.dropout(hidden)
        )
        return predictions, hidden, cell, alpha

    # Generate greedy caption
    def generate_caption(
        self,
        encoder_features,
        vocab,
        max_length=20
    ):

        batch_size = encoder_features.size(0)
    
        h = torch.zeros(
            batch_size,
            self.lstm.hidden_size,
            device=encoder_features.device
        )
    
        c = torch.zeros_like(h)
    
        current_word = torch.tensor(
            [vocab.word2idx["<start>"]],
            device=encoder_features.device
        )
    
        generated_caption = []
        attention_weights = []
    
        for _ in range(max_length):
            prediction, h, c, alpha = self.decode_step(
                encoder_features,
                current_word,
                h,
                c
            )
    
            predicted_word = prediction.argmax(dim=1)
            word = vocab.idx2word[
                predicted_word.item()
            ]
    
            if word == "<end>":
                break
    
            generated_caption.append(word)
            attention_weights.append(alpha)
            current_word = predicted_word
        return generated_caption, attention_weights

        
    
    # using beam search
    def generate_caption_beam_search(
        self,
        encoder_features,
        vocab,
        beam_size=3,
        max_length=20
    ):

        batch_size = encoder_features.size(0)
    
        # Initialize hidden and cell states
        h = torch.zeros(
            batch_size,
            self.lstm.hidden_size,
            device=encoder_features.device
        )
    
        c = torch.zeros_like(h)
    
        # Initial beam
        beams = [{
            "caption": [vocab.word2idx["<start>"]],
            "hidden": h,
            "cell": c,
            "score": 0.0,
            "attention": []
        }]
    
        # Generate words
        for _ in range(max_length):
    
            new_beams = []
    
            # Expand each current beam
            for beam in beams:
    
                caption = beam["caption"]
                hidden = beam["hidden"]
                cell = beam["cell"]
                score = beam["score"]
                attention = beam["attention"]
    
                # If this beam already ended keep it
                if caption[-1] == vocab.word2idx["<end>"]:
                    new_beams.append(beam)
                    continue
    
                # Current input word
                current_word = torch.tensor(
                    [[caption[-1]]], # as shape is 1,1
                    device=encoder_features.device
                )
    
                # Decode one step
                predictions, hidden, cell, alpha = self.decode_step(
                    encoder_features,
                    current_word,
                    hidden,
                    cell
                )
    
                # Convert logits to log probabilities
                log_probs = F.log_softmax(
                    predictions,
                    dim=1
                )
    
                # Keep top beam_size candidate words
                top_scores, top_words = log_probs.topk(
                    beam_size,
                    dim=1
                )
    
                # Create child beam for each candidate
                for i in range(beam_size):
    
                    next_word = top_words[0, i].item()
                    next_score = top_scores[0, i].item()
    
                    new_caption = caption + [next_word]
                    new_score = score + next_score
                    new_attention = attention + [alpha]
    
                    new_beams.append({
                        "caption": new_caption,
                        "hidden": hidden.clone(),
                        "cell": cell.clone(),
                        "score": new_score,
                        "attention": new_attention
                    })
    
            # Keep only the best beams
            new_beams = sorted(
                new_beams,
                key=lambda beam: beam["score"],
                reverse=True
            )
    
            beams = new_beams[:beam_size]
    
        # Best beam
        best_beam = beams[0]
    
        generated_caption = []
    
        # Convert token IDs to words
        for token in best_beam["caption"]:
    
            if token == vocab.word2idx["<start>"]:
                continue
    
            if token == vocab.word2idx["<end>"]:
                break
    
            generated_caption.append(
                vocab.idx2word[token]
            )
    
        return generated_caption, best_beam["attention"]




# HyperParameters
checkpoint = torch.load("vocab.pth", map_location=device)

vocab = Vocabulary(min_freq=checkpoint["min_freq"])
vocab.word2idx = checkpoint["word2idx"]
vocab.idx2word = checkpoint["idx2word"]


embed_size = 256
hidden_size = 512
attention_dim = 256
encoder_dim = 2048
vocab_size = len(vocab)
num_epochs = 10

# Model Creation
encoder = EncoderCNN().to(device)
attention = BahdanauAttention(
    encoder_dim = encoder_dim,
    decoder_dim = hidden_size,
    attention_dim = attention_dim
)

decoder = DecoderWithAttention(
    attention = attention,
    embed_size = embed_size,
    hidden_size = hidden_size,
    vocab_size = vocab_size,
    encoder_dim = encoder_dim,
    dropout = 0.5
).to(device)



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_uploaded_image(
    image_path,
    beam_search=True,
    beam_size=3,
    show_attention=False
):
    """
    Generate caption for uploaded image
    """
    encoder.eval()
    decoder.eval()

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Preprocessing
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_features = encoder(image_tensor)

        if beam_search:
            caption, attention = decoder.generate_caption_beam_search(
                encoder_features,
                vocab,
                beam_size=beam_size
            )
        else:
            caption, attention = decoder.generate_caption(
                encoder_features,
                vocab
            )

    # Image for display
    display_image = image_tensor.squeeze(0).cpu()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    display_image = display_image * std + mean
    display_image = display_image.clamp(0, 1)
    display_image = display_image.permute(1, 2, 0).numpy()

    if not show_attention:
        return " ".join(caption)

    stop_words = {
        "a", "an", "the", "is", "are",
        "on", "in", "at", "of", "to",
        "with", "and", "by", "for",
        "from", "into", "over", "under"
    }

    plot_words = []
    plot_attention = []

    for word, alpha in zip(caption, attention):
        if word.lower() in stop_words or word == ".":
            continue

        plot_words.append(word)
        plot_attention.append(alpha.cpu())

    return " ".join(caption), display_image, plot_words, plot_attention

def load_checkpoint(
    checkpoint_path,
    encoder,
    decoder,
    optimizer = None,
    device = "cuda"
):
    """ 
    Getting back all the modle parameters 
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location = device
    )
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)

    print("Checkpoint Loaded successfully!")
    print(f"Loaded Checkpoint Path : {checkpoint_path}")
    print(f"Epoch : {epoch}")

def create_models(
    vocab_size,
    device = "cuda",
    embed_size = 256,
    hidden_size = 512,
    attention_dim = 256,
    encoder_dim = 2048,
    num_epochs = 10
):
    encoder = EncoderCNN().to(device)
    attention = BahdanauAttention(
        encoder_dim = encoder_dim,
        decoder_dim = hidden_size,
        attention_dim = attention_dim
    ).to(device)

    decoder = DecoderWithAttention(
        attention = attention,
        embed_size = embed_size,
        hidden_size = hidden_size,
        vocab_size = vocab_size,
        encoder_dim = encoder_dim,
        dropout = 0.5
    ).to(device)

    return encoder, attention, decoder

def get_models(
    vocab,
    device = "cuda",
    checkpoint_path = None
):
    encoder, attention, decoder = create_models(
        vocab_size = len(vocab),
        device = device
    )
    start_epoch = 0
    loss = None
    if checkpoint_path is not None:
        load_checkpoint(
            checkpoint_path = checkpoint_path,
            encoder = encoder,
            decoder = decoder,
            device = device
        )
    return encoder, attention, decoder

encoder, attention, decoder = get_models(
    vocab=vocab,
    device=device,
    checkpoint_path = "best_model.pth"
)

encoder.eval()
decoder.eval()
