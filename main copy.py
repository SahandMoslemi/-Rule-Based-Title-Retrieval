import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import Masoud2
from config import MASOUD_1, MASOUD_6, MASOUD_7, MASOUD_8, MASOUD_9

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, output, target):
        output_norm = output / output.norm(dim=1, keepdim=True)
        target_norm = target / target.norm(dim=1, keepdim=True)
        cosine_sim = (output_norm * target_norm).sum(dim=1)
        loss = 1 - cosine_sim.mean()
        return loss

class EmbeddingRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),  
            nn.ReLU(),
            nn.Linear(1024, 512),  
            nn.ReLU(),
            nn.Linear(512, 256),  
            nn.ReLU(),
            nn.Linear(256, 512),  
            nn.ReLU(),
            nn.Linear(512, output_dim)   
        )

    def forward(self, x):
        return self.network(x)

class ModelTrainer:
    def __init__(self, model, learning_rate, epochs):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.criterion = CosineSimilarityLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self, train_inputs, train_targets):
        for epoch in range(self.epochs):
            outputs = self.model(train_inputs)
            loss = self.criterion(outputs, train_targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')

class InferenceEngine:
    @staticmethod
    def infer(model, tag_embedding):
        model.eval()
        with torch.no_grad():
            predicted_title_embedding = model(tag_embedding)
        return predicted_title_embedding

    @staticmethod
    def find_closest(predicted_embedding, title_embeddings, titles):
        similarities = cosine_similarity(predicted_embedding.numpy(), title_embeddings)
        closest_idx = np.argmax(similarities)
        return titles[closest_idx], similarities[0][closest_idx]

class DatasetLoader:
    def __init__(self, masoud_args):
        self.df = Masoud2(*masoud_args).masoud_3

    def prepare_data(self):
        X = np.stack(self.df['avg_tags_embedding'].values)
        y = np.stack(self.df['avg_title_embedding'].values)
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, np.arange(len(self.df)), test_size=0.2, random_state=42)

        return torch.FloatTensor(X_train), torch.FloatTensor(X_test), torch.FloatTensor(y_train), torch.FloatTensor(y_test), train_indices, test_indices

if __name__ == "__main__":
    # Configuration
    masoud_args = ("bharatkumar0925/tmdb-movies-clean-dataset", MASOUD_1, MASOUD_6, "leadbest/googlenewsvectorsnegative300", MASOUD_7, MASOUD_8, MASOUD_9)
    hidden_dim = 128
    learning_rate = 0.01
    epochs = 500

    # Data Loading
    dataset_loader = DatasetLoader(masoud_args)
    X_train, X_test, y_train, y_test, X_train_indices, X_test_indices = dataset_loader.prepare_data()

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    # Model Initialization
    model = EmbeddingRegressor(input_dim, hidden_dim, output_dim)
    trainer = ModelTrainer(model, learning_rate, epochs)

    # Training
    trainer.train(X_train, y_train)

    #
    # Inference
    sample_tag = X_train[0].unsqueeze(0)
    predicted_embedding = InferenceEngine.infer(model, sample_tag)

    # Closest Title Search
    title_embeddings = np.stack(dataset_loader.df['avg_title_embedding'].values)
    closest_title, similarity = InferenceEngine.find_closest(predicted_embedding, title_embeddings, dataset_loader.df['title'].values)

    # Results
    print(f"Input tags: {dataset_loader.df['tags'].iloc[X_train_indices[0]]}")
    print(f"Actual title: {dataset_loader.df['title'].iloc[X_train_indices[0]]}")
    print(f"Predicted embedding: {predicted_embedding.numpy().flatten()[:5]}...")
    print(f"Closest title predicted: {closest_title}")
    print(f"Similarity score: {similarity:.4f}")
