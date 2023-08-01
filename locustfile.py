from locust import HttpUser, task, between
from locust import events as ev
import pinecone
from dotenv import load_dotenv
import os
import time
from datasets import load_dataset
import torch
import clip
import random

pinecone.init(api_key=os.environ['PINECONE_API_KEY'])
index = pinecone.Index(os.environ['PINECONE_INDEX_NAME'])
top_k = 10

# load 100 random images from test dataset
test_dataset = load_dataset("fashion_mnist")['test'].shuffle().select(range(0, 100))

# Check to see if GPU is aviailable
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

label_descriptions = {0: "T-shirt/top", 
           1: "Trouser",
           2: "Pullover",
           3: "Dress",
           4: "Coat",
           5: "Sandal",
           6: "Shirt",
           7: "Sneaker",
           8: "Bag",
           9: "Ankle boot"}

# Generate vector embeddings for each image in the dataset
test_vectors = []
for image in test_dataset:
    with torch.no_grad():
        image_pp = preprocess(image['image']).unsqueeze(0).to(device)
        image_features = model.encode_image(image_pp)
        embedding_numpy = image_features.cpu().numpy().squeeze().tolist()
        test_vectors.append({'embedding': embedding_numpy,
                        'description': label_descriptions[image["label"]]})

class PineconeUser(HttpUser):
    
    wait_time = between(0, 1)  # Define the wait time between consecutive tasks for a user

    @task
    def query_pinecone(self):

        top_k_hit = False

        # Select a random image+embedding from the test dataset
        test_vector = random.choice(test_vectors)

        start_time = time.time()
        query_result = index.query(
            vector = test_vector['embedding'],
            namespace="fashion-mnist",
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        end_time = time.time()

        response_time = (end_time - start_time) * 1000
        for match in query_result.matches:
            if match.metadata['description'] == test_vector['description']:
                top_k_hit = True
        
        if top_k_hit == True:
            ev.request.fire(request_type="grpc",
                        name="query_pinecone",
                        response_time=response_time,
                        response_length=top_k,
                    )
        else:
            ev.request.fire(request_type="grpc",
                        name="query_pinecone",
                        response_time=response_time,
                        response_length=top_k,
                        exception="Top K results do not contain the correct label"
                    )
        