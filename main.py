from preprocessing import Masoud2
from config import MASOUD_1, MASOUD_6, MASOUD_7, MASOUD_8, MASOUD_9
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity


def average_embedding(embedding_list):
    return np.mean(embedding_list, axis=0)

class EmbeddingRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def train_model(model, train_inputs, train_targets, learning_rate, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        outputs = model(train_inputs)
        loss = criterion(outputs, train_targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def inference(model, tag_embedding):
    model.eval()
    with torch.no_grad():
        predicted_title_embedding = model(tag_embedding)
    return predicted_title_embedding

def find_closest_title(predicted_embedding, title_embeddings, titles):
    similarities = cosine_similarity(predicted_embedding.numpy(), title_embeddings)
    closest_idx = np.argmax(similarities)
    return titles[closest_idx], similarities[0][closest_idx]

if __name__ == "__main__":
    df = Masoud2("bharatkumar0925/tmdb-movies-clean-dataset", MASOUD_1, MASOUD_6, "leadbest/googlenewsvectorsnegative300", MASOUD_7, MASOUD_8, MASOUD_9).masoud_3

    df['avg_title_embedding'] = df['title_embedding'].apply(average_embedding)
    df['avg_tags_embedding'] = df['tags_embedding'].apply(average_embedding)
    
    X = np.stack(df['avg_tags_embedding'].values)
    y = np.stack(df['avg_title_embedding'].values)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    indices = np.arange(len(df))
    _, X_test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    input_dim = X_train.shape[1]
    hidden_dim = 128
    output_dim = y_train.shape[1]
    
    model = EmbeddingRegressor(input_dim, hidden_dim, output_dim)
    
    learning_rate = 0.01
    epochs = 1000
    train_model(model, X_train, y_train, learning_rate, epochs)
    
    sample_tag = X_test[0].unsqueeze(0)
    predicted_embedding = inference(model, sample_tag)
    
    title_embeddings = np.stack(df['avg_title_embedding'].values)
    closest_title, similarity = find_closest_title(predicted_embedding, title_embeddings, df['title'].values)
    
    print(f"Input tags: {df['tags'].iloc[X_test_indices[0]]}")
    print(f"Actual title: {df['title'].iloc[X_test_indices[0]]}")
    print(f"Predicted embedding: {predicted_embedding.numpy().flatten()[:5]}...")
    print(f"Closest title predicted: {closest_title}")
    print(f"Similarity score: {similarity:.4f}")