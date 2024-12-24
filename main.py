from preprocessing import Masoud2
from config import MASOUD_1, MASOUD_6, MASOUD_7, MASOUD_8, MASOUD_9
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
import logging
import matplotlib.pyplot as plt

def find_n_closest_documents_l1(predicted_embedding, title_embeddings, titles, n):
    distances = np.sum(np.abs(title_embeddings - predicted_embedding.numpy()), axis=1)
    sorted_indices = np.argsort(distances)[:n]
    return [(titles[idx], distances[idx]) for idx in sorted_indices]

def calculate_precision_at_k(retrieved_titles, relevant_titles, k):
    top_k_retrieved = retrieved_titles[:k]
    relevant_in_top_k = sum(1 for title in top_k_retrieved if title in relevant_titles)
    return relevant_in_top_k / k



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

def embedding_output(df, dataclass, output_path):
    embeddings_df = pd.DataFrame(df[dataclass].to_list())
    embeddings_df.to_csv(output_path, index=False)



if __name__ == "__main__":
    df = Masoud2("bharatkumar0925/tmdb-movies-clean-dataset", MASOUD_1, MASOUD_6, "leadbest/googlenewsvectorsnegative300", MASOUD_7, MASOUD_8, MASOUD_9).masoud_3
    df = df.sample(n=1000, random_state=42)

    embeddings = {
        'Average': ('avg_tags_embedding', 'avg_title_embedding'),
        'Max': ('max_tags_embedding', 'max_title_embedding')
    }

    precision_at_k_results = {key: {5: [], 10: [], 15: [], 20: []} for key in embeddings.keys()}

    for embedding_type, (tags_column, titles_column) in embeddings.items():
        X = np.stack(df[tags_column].values)
        y = np.stack(df[titles_column].values)

        X_train, X_test, y_train, y_test, X_train_indices, X_test_indices = train_test_split(X, y, np.arange(len(df)), test_size=0.2, random_state=42)

        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)

        input_dim = output_dim = X_train.shape[1]

        # Train
        model = EmbeddingRegressor(input_dim, output_dim)

        learning_rate = 1
        epochs = 5000
        train_model(model, X_train, y_train, learning_rate, epochs)

        # Precision@K 
        for i in range(len(X_train)):
            sample_tag = X_train[i].unsqueeze(0)
            predicted_embedding = inference(model, sample_tag)

            closest_documents = find_n_closest_documents_l1(predicted_embedding, y, df['title'].values, max(precision_at_k_results[embedding_type].keys()))
            retrieved_titles = [doc[0] for doc in closest_documents]

            # ground truth titles for data
            relevant_titles = [df['title'].iloc[X_train_indices[i]]]

            for k in precision_at_k_results[embedding_type].keys():
                precision = calculate_precision_at_k(retrieved_titles, relevant_titles, k)
                precision_at_k_results[embedding_type][k].append(precision)

    # mean precision for k
    mean_precision_at_k_results = {
        embedding_type: {k: np.mean(precision_at_k_results[embedding_type][k]) for k in precision_at_k_results[embedding_type].keys()}
        for embedding_type in embeddings.keys()
    }

    # Plot Precision@K for both avg and max
    plt.figure()
    for embedding_type, mean_precision_at_k in mean_precision_at_k_results.items():
        plt.plot(mean_precision_at_k.keys(), mean_precision_at_k.values(), marker='o', label=f'{embedding_type} embeddings')

    plt.title('Precision@K Comparison')
    plt.xlabel('K')
    plt.ylabel('Mean Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig('precision_at_k_comparison_plot.png')

