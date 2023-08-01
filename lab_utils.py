import numpy as np
import random
import time
import requests
import shutil
import gzip
import torch
from PIL import Image
import torchvision.transforms as transforms
import clip
import itertools


def generate_vectors(dimensions):
    vectors = []
    id_seed = 1
    value_seed = 0.1

    for _ in range(500):
        meta_data = {"category": random.choice(["one", "two", "three"]),
                     "timestamp": time.time()}
        embeddings = np.full(shape=dimensions, fill_value=value_seed).tolist()
        vectors.append({'id': str(id_seed),
                        'values': embeddings,
                        'metadata': meta_data})
        id_seed = id_seed + 1
        value_seed = value_seed + 0.1
    return vectors

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

def image_to_embedding(image_path):
    # Load and preprocess the image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # Generate the image features
    with torch.no_grad():
        image_features = model.encode_image(image)
        
    return image_features

def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

