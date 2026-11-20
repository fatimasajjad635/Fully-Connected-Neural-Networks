#!/usr/bin/env python3
"""
MSE 641 Assignment 4: Neural Network Classifier with PyTorch
-----------------------------------------
Author: [Fatima Sajjad]
Student ID: [20718749]

This script implements a fully-connected feed-forward neural network classifier
for sentiment analysis of Amazon reviews using Word2Vec embeddings.
"""
import csv
import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def load_embeddings(embeddings_path):
    """
    Load Word2Vec embeddings from Assignment 3
    
    Args:
        embeddings_path: Path to the Word2Vec model file
        
    Returns:
        word2vec_model: Loaded Word2Vec model
    """
    # TODO: Implement this function
    # Load your Word2Vec model from Assignment 3
    # Handle both .pkl and .model file formats
    # Return the loaded model
    
    if embeddings_path.endswith(".model"):
        word2vec_model = Word2Vec.load(embeddings_path)
    elif embeddings_path.endswith(".pkl"):
        with open(embeddings_path, "rb") as f:
            word2vec_model = pickle.load(f)
    else:
        raise ValueError("Unsupported file format. Use .model or .pkl")
    
    return word2vec_model

def load_data_and_labels(data_file, labels_file):
    """
    Load tokenized data and labels from CSV files
    
    Args:
        data_file: Path to tokenized data file
        labels_file: Path to labels file
        
    Returns:
        tuple: (processed_data, processed_labels)
    """
    # TODO: Implement this function
    # Load and process your data from Assignment 1
    # Ensure data and labels are properly aligned
    # Return tokenized documents and corresponding labels
    
    data_df = pd.read_csv(data_file, on_bad_lines='skip')
    labels_df = pd.read_csv(labels_file)

    data_df = pd.read_csv(data_file, on_bad_lines='skip')
    labels_df = pd.read_csv(labels_file)

    processed_data = data_df.iloc[:, 0].astype(str).apply(lambda x: x.split(',')).tolist()
    processed_labels = labels_df.iloc[:, 0].tolist()

    return processed_data, processed_labels
    """
    Convert tokenized documents to document embeddings using Word2Vec
    
    Args:
        tokenized_documents: List of tokenized documents (list of lists)
        word2vec_model: Trained Word2Vec model
        
    Returns:
        numpy.ndarray: Document embeddings matrix
    """
    # TODO: Implement this function
    # For each document, compute the average of word embeddings
    # Handle out-of-vocabulary words appropriately
    # Return a matrix where each row is a document embedding
    
    document_embeddings = []

    for doc in tokenized_documents:
        vectors = [word2vec_model.wv[word] for word in doc if word in word2vec_model.wv]
        
        if vectors:
            avg_vector = np.mean(vectors, axis=0)
        else:
            avg_vector = np.zeros(word2vec_model.vector_size)

        document_embeddings.append(avg_vector)

    return np.array(document_embeddings)

    """
    Fully-connected neural network classifier for sentiment analysis
    """

    """
        Initialize the neural network
        
        Args:
            input_size: Size of input features (Word2Vec embedding dimension)
            hidden_size: Size of hidden layer
            num_classes: Number of output classes
            activation: Activation function ('relu', 'sigmoid', 'tanh')
            dropout_rate: Dropout rate for regularization
        """
class NeuralNetworkClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation='relu', dropout_rate=0.5):
        super(NeuralNetworkClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Invalid activation function. Choose from 'relu', 'sigmoid', 'tanh'.")

        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output logits
        """
        # TODO: Implement the forward pass
        # 1. Input to hidden layer
        # 2. Apply activation function
        # 3. Apply dropout
        # 4. Hidden to output layer
        # 5. Return output logits (no softmax, as CrossEntropyLoss includes it)
        
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, l2_reg=0.001):
    """
    Train the neural network model
    
    Args:
        model: Neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        l2_reg: L2 regularization strength
        
    Returns:
        dict: Training history with losses and accuracies, best validation accuracy
    """
    # TODO: Implement the training loop
    # 1. Set up loss function (CrossEntropyLoss)
    # 2. Set up optimizer (Adam with L2 regularization via weight_decay)
    # 3. Training loop with validation monitoring
    # 4. Track training and validation losses/accuracies
    # 5. Implement early stopping based on validation performance
    # 6. Return training history and best validation accuracy
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'best_val_acc': 0.0
        }

    best_model_state = None
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        correct, total, train_loss = 0, 0, 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        correct, total, val_loss = 0, 0, 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        history['best_val_acc'] = best_val_acc
        return history

def evaluate_model(model, test_loader):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained neural network model
        test_loader: DataLoader for test data
        
    Returns:
        float: Test accuracy
    """
    # TODO: Implement model evaluation
    # 1. Set model to evaluation mode
    # 2. Disable gradient computation
    # 3. Make predictions on test data
    # 4. Calculate and return accuracy
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    test_accuracy = correct / total
    return test_accuracy

def make_sure_dir_exists(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def create_document_embeddings(tokenized_documents, word2vec_model):
    document_embeddings = []

    for doc in tokenized_documents:
        vectors = [word2vec_model.wv[word] for word in doc if word in word2vec_model.wv]

        if vectors:
            avg_vector = np.mean(vectors, axis=0)
        else:
            avg_vector = np.zeros(word2vec_model.vector_size)

        document_embeddings.append(avg_vector)

    return np.array(document_embeddings)

def main():
    """
    Main execution function
    """
    # TODO: Implement the main execution flow
    # 1. Load Word2Vec embeddings from Assignment 3
    # 2. Load and prepare data (train/val/test splits)
    # 3. Create document embeddings
    # 4. Set up data loaders
    # 5. Define experimental configurations
    # 6. Train models with different settings
    # 7. Evaluate and save results
    # 8. Save the best model
    
    print("MSCI 641 Assignment 4: Neural Network Classifier")
    print("=" * 50)
    
    # Configuration
    EMBEDDING_PATH = "embeddings/word2vec_model.pkl"  # Adjust path as needed
    DATA_DIR = "data"
    MODELS_DIR = "models"
    
    # Hyperparameters
    HIDDEN_SIZE = 128
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Experimental configurations
    experiments = [
        # (activation, l2_reg, dropout_rate)
        ('relu', 0.001, 0.3),
        ('relu', 0.01, 0.3),
        ('relu', 0.001, 0.5),
        ('sigmoid', 0.001, 0.3),
        ('sigmoid', 0.01, 0.3),
        ('tanh', 0.001, 0.3),
        ('tanh', 0.01, 0.3),
    ]

    # Create models directory
    make_sure_dir_exists(MODELS_DIR)
    
    # TODO: Implement the rest of the main function
    # Follow the steps outlined in the docstring above
    word2vec_model = load_embeddings(EMBEDDING_PATH)

    train_data, train_labels = load_data_and_labels(f"{DATA_DIR}/train_subset.csv", f"{DATA_DIR}/train_labels_subset.csv")
    val_data, val_labels = load_data_and_labels(f"{DATA_DIR}/val_subset.csv", f"{DATA_DIR}/val_labels_subset.csv")
    test_data, test_labels = load_data_and_labels(f"{DATA_DIR}/test.csv", f"{DATA_DIR}/test_labels.csv")
    
    label_map = {"negative": 0, "positive": 1}
    train_labels = [label_map[label.lower()] for label in train_labels]
    val_labels = [label_map[label.lower()] for label in val_labels]
    test_labels = [label_map[label.lower()] for label in test_labels]

    X_train = create_document_embeddings(train_data, word2vec_model)
    X_val = create_document_embeddings(val_data, word2vec_model)
    X_test = create_document_embeddings(test_data, word2vec_model)

    train_labels = train_labels[:len(X_train)]
    val_labels = val_labels[:len(X_val)]
    test_labels = test_labels[:len(X_test)]

    import torch
    from torch.utils.data import TensorDataset, DataLoader

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(train_labels)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val).float(), torch.tensor(val_labels)), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test).float(), torch.tensor(test_labels)), batch_size=BATCH_SIZE)

    best_accuracy = 0.0
    best_model = None
   
    word2vec_model = load_embeddings(EMBEDDING_PATH)

    train_data, train_labels = load_data_and_labels(f"{DATA_DIR}/train_subset.csv", f"{DATA_DIR}/train_labels_subset.csv")
    val_data, val_labels = load_data_and_labels(f"{DATA_DIR}/val_subset.csv", f"{DATA_DIR}/val_labels_subset.csv")
    test_data, test_labels = load_data_and_labels(f"{DATA_DIR}/test.csv", f"{DATA_DIR}/test_labels.csv")

    X_train = create_document_embeddings(train_data, word2vec_model)
    X_val = create_document_embeddings(val_data, word2vec_model)
    X_test = create_document_embeddings(test_data, word2vec_model)

    activation_best_models = {}
    activation_best_accuracies = {}
    activation_best_settings = {}
   
    with open("hyperparameter_tuning.csv", mode="w", newline="") as tuning_file, \
        open("results.csv", mode="w", newline="") as results_file:

        tuning_writer = csv.writer(tuning_file)
        results_writer = csv.writer(results_file)

        tuning_writer.writerow(["Activation Function", "L2 Regularization", "Dropout Rate", "Validation Accuracy"])
        results_writer.writerow(["Activation Function", "L2 Regularization", "Dropout Rate", "Validation Accuracy", "Test Accuracy"])

        for activation, l2, dropout in experiments:
            print(f"\nTraining with: activation={activation}, l2_reg={l2}, dropout={dropout}")

            model = NeuralNetworkClassifier(
                input_size=X_train.shape[1],
                hidden_size=HIDDEN_SIZE,
                num_classes=2,
                activation=activation,
                dropout_rate=dropout
            )

            history = train_model(model, train_loader, val_loader,
                                num_epochs=NUM_EPOCHS,
                                learning_rate=LEARNING_RATE,
                                l2_reg=l2)

            val_acc = history['best_val_acc']
            print(f"Best Validation Accuracy: {val_acc:.4f}")

            tuning_writer.writerow([activation, l2, dropout, val_acc])

            if activation not in activation_best_accuracies or val_acc > activation_best_accuracies[activation]:
                activation_best_accuracies[activation] = val_acc
                activation_best_models[activation] = model
                activation_best_settings[activation] = (l2, dropout)
                    
        for activation in activation_best_models:
            l2, dropout = activation_best_settings[activation]
            model = activation_best_models[activation]
            val_acc = activation_best_accuracies[activation]
            test_acc = evaluate_model(model, test_loader)
            results_writer.writerow([activation, l2, dropout, val_acc, test_acc])

if __name__ == "__main__":
    main()