from preprocessing import Masoud2
from config import MASOUD_1, MASOUD_6, MASOUD_7, MASOUD_8, MASOUD_9
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
import logging


class EmbeddingRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def train_model(model, train_inputs, train_targets, learning_rate, epochs, regularization_lambda=1e-4):
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        outputs = model(train_inputs)
        loss = criterion(outputs, train_targets)
        
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss += regularization_lambda * l2_norm
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            logging.info(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def inference(model, tag_embedding):
    model.eval()
    with torch.no_grad():
        predicted_title_embedding = model(tag_embedding)
    return predicted_title_embedding

def find_closest_title_cosine(predicted_embedding, title_embeddings, titles):
    similarities = cosine_similarity(predicted_embedding.numpy(), title_embeddings)
    closest_idx = np.argmax(similarities)
    return titles[closest_idx], similarities[0][closest_idx]

def find_closest_title_l1(predicted_embedding, title_embeddings, titles):
    distances = np.sum(np.abs(title_embeddings - predicted_embedding.numpy()), axis=1)
    closest_idx = np.argmin(distances)
    return titles[closest_idx], distances[closest_idx]

if __name__ == "__main__":
    df = Masoud2("bharatkumar0925/tmdb-movies-clean-dataset", MASOUD_1, MASOUD_6, "leadbest/googlenewsvectorsnegative300", MASOUD_7, MASOUD_8, MASOUD_9).masoud_3
    df = df.sample(n=20, random_state=42)
    
    X = np.stack(df['avg_tags_embedding'].values)
    y = np.stack(df['avg_title_embedding'].values)

    X_train, X_test, y_train, y_test, X_train_indices, X_test_indices = train_test_split(X, y, np.arange(len(df)), test_size=0.2, random_state=42)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    input_dim = output_dim = X_train.shape[1]
    
    # Train
    model = EmbeddingRegressor(input_dim, output_dim)
    
    learning_rate = 1
    epochs = 1000
    train_model(model, X_train, y_train, learning_rate, epochs)

    # Inference
    for i in range(10):
        sample_tag = X_train[i].unsqueeze(0)

        predicted_embedding = inference(model, sample_tag)
        
        closest_title, distance = find_closest_title_l1(predicted_embedding, y, df['title'].values)

        logging.info("=====================================")
        logging.info(f"Input tags: {df['tags'].iloc[X_train_indices[i]]}")
        logging.info(f"Actual title: {df['title'].iloc[X_train_indices[i]]}")
        logging.info(f"Predicted embedding: {predicted_embedding.numpy().flatten()[:5]}...")
        logging.info(f"Closest title predicted: {closest_title}")
        logging.info(f"L1 Distance score: {distance:.4f}")

        sample_tag = X_train[i].unsqueeze(0)

        predicted_embedding = inference(model, sample_tag)
        
        closest_title, distance = find_closest_title_cosine(predicted_embedding, y, df['title'].values)
        
        logging.info(f"Input tags: {df['tags'].iloc[X_train_indices[i]]}")
        logging.info(f"Actual title: {df['title'].iloc[X_train_indices[i]]}")
        logging.info(f"Predicted embedding: {predicted_embedding.numpy().flatten()[:5]}...")
        logging.info(f"Closest title predicted: {closest_title}")
        logging.info(f"Similarity score: {distance:.4f}")
        logging.info("=====================================")
