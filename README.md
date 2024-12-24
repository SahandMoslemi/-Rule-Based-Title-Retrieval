# Project Title: **Content-based Movie Title Retrieval**
## Overview
This project explores the development of a hybrid content-based retrieval system that integrates deep learning models with rule-based representation learning. The aim is to balance the trade-off between interpretability and performance in recommendation systems.

Traditional methods, while transparent, face challenges in scalability and handling complex data distributions. On the other hand, deep learning excels at feature extraction and retrieval efficiency but often lacks interpretability. To address these limitations, this project incorporates logical layers informed by knowledge graphs into deep learning architectures. By combining these approaches, the system leverages the interpretability of rule-based models and the performance benefits of deep learning.

Using the TMDB Movies Clean dataset with over 660,000 records, the project implements a robust preprocessing pipeline that includes tokenization and embedding generation using Word2Vec. A Rule Representation Learner (RRL) discretizes continuous features, while Gradient Grafting integrates discrete and continuous gradients during model training.

Experimental results highlight the effectiveness of this hybrid approach, demonstrating improvements in retrieval performance and generalization capabilities. Precision@K metrics consistently outperform baseline embeddings, with the integration of knowledge graphs showing improved semantic understanding and interpretability through explicit rule generation. This project offers a promising direction for building efficient and transparent content-based retrieval systems.

---

## Features
- **Text Embedding and Preprocessing**:
  - Automated dataset download, cleaning, and embedding generation.
  - Pre-trained Word2Vec integration for numerical representations of text.
- **Embedding Regression**:
  - Deep neural network for embedding prediction.
  - Precision@K evaluation for relevance ranking.
- **LLM Fine-Tuning**:
  - Fine-tunes PyThea, a large language model, on domain-specific text data.
  - Enables custom natural language understanding and generation tasks.
- **Logging and Visualization**:
  - Comprehensive logging for debugging and progress tracking.
  - Precision@K comparison plots for evaluating embedding performance.

---

## Prerequisites
Install the required Python packages using:
```bash
pip install -r requirements.txt
```

---

## Configuration
Refer to `config.py` for all file paths, model paths, and logging configurations.

---

## LLM Fine-Tuning

### Notebook: `llm.ipynb`
This Jupyter notebook demonstrates the fine-tuning process for PyThea, a large language model, on a specific text dataset.

### Features
- **Dataset Preparation**:
  - Loads and preprocesses text data for fine-tuning.
- **Model Configuration**:
  - Sets up the PyThea model and tokenizer for domain-specific fine-tuning.
- **Training Pipeline**:
  - Trains the model with hyperparameters optimized for text generation tasks.
- **Evaluation**:
  - Generates text samples to evaluate the model's performance.
  - Saves the fine-tuned model for deployment.

### Usage
1. Open the notebook in JupyterLab or Jupyter Notebook:
   ```bash
   jupyter notebook llm.ipynb
   ```
2. Follow the step-by-step instructions to:
   - Load the dataset.
   - Fine-tune the PyThea model.
   - Evaluate and save the fine-tuned model.

### Outputs
- **Fine-Tuned Model**: Saved to the `models/` directory.
- **Generated Text Samples**: Output directly in the notebook for quick evaluation.

---

## Usage

### Text Embedding
Run `preprocessing.py` to preprocess and generate embeddings:
```bash
python preprocessing.py
```

### Embedding Regression and Evaluation
Run `main.py` to train the embedding regression model and evaluate relevance:
```bash
python main.py
```

### Fine-Tuning PyThea
Open and execute the `llm.ipynb` notebook for LLM fine-tuning.

---

## File Structure
```
project/
├── preprocessing.py       # Core script for preprocessing and embedding
├── main.py                # Embedding regression and evaluation
├── llm.ipynb              # Jupyter Notebook for LLM fine-tuning (PyThea)
├── config.py              # Centralized configuration management
├── models/                # Directory for storing models
├── data/                  # Directory for datasets
├── tmp/                   # Temporary files and logs
│   └── logs/              # Log files
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## Example Dataset
- Kaggle Dataset: [TMDB Movies Clean Dataset](https://www.kaggle.com/datasets/bharatkumar0925/tmdb-movies-clean-dataset)

---

## Contributing
Contributions are welcome! Fork the repository and submit a pull request with enhancements or bug fixes.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
