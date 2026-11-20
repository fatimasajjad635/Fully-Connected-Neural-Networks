Implementing and training fully-connected neural network classifiers using PyTorch on the Amazon reviews dataset and submitting your trained models.

## Overview

In this assignment, you will:

1. Implement a fully-connected feed-forward neural network classifier using PyTorch
2. Train models with different activation functions (ReLU, sigmoid, tanh)
3. Experiment with L2 regularization and dropout
4. Evaluate the models on your test set
5. Upload your trained models to the university server
6. Submit your code and results to GitHub Classroom

## Prerequisites

- Completed Assignment 1 (data preparation), Assignment 2 (Naive Bayes), and Assignment 3 (Word2Vec)
- Python 3.10+ with PyTorch, scikit-learn, numpy, gensim and pandas installed
- An API key for the model server (provided by the instructor). A unique API key was generated for each student and is available on Learn > Grades > API Key.

### Opening in Codespaces

1. Navigate to your assignment repository on GitHub
2. Click the green "Code" button
3. Select the "Codespaces" tab
4. Click "Create codespace on main"

## Step 1: Getting Your Data and Word2Vec Embeddings into GitHub Codespaces

To bring your preprocessed data from Assignment 1 and Word2Vec embeddings from Assignment 3 into GitHub Codespaces for Assignment 4, follow these steps:

1. Create directories for your data:
   ```bash
   mkdir -p data
   mkdir -p embeddings
   ```

2. Click the Explorer icon in the left sidebar

3. Upload the following preprocessed data files from Assignment 1 to the `data` directory:
   - `train.csv` and `train_labels.csv`
   - `val.csv` and `val_labels.csv` 
   - `test.csv` and `test_labels.csv`

4. Upload your Word2Vec embeddings to the `embeddings` directory:
   - Right-click on the `embeddings` directory and select "Upload..."
   - Upload your Word2Vec model file from Assignment 3 (e.g., `word2vec_model.pkl` or `word2vec_model.model`)

Note: The `data` and `embeddings` directories contain large files and have been added to .gitignore, so they will not be uploaded to GitHub.

### ⚠️ Important: Data Size Considerations

**The full training set (`train.csv` or `train_ns.csv`) is too large for Codespaces.** You should use a subset for development and testing:

#### Option 1: Create Subsets of `train` and `val` Datasets

Copy and paste these commands in the terminal.

```bash
# Create a 100K line subset for faster training
head -100000 data/train.csv > data/train_subset.csv
head -100000 data/train_labels.csv > data/train_labels_subset.csv

# select 10K line subset for validation
head -10000 data/val.csv > data/val_subset.csv
head -10000 data/val_labels.csv > data/val_labels_subset.csv

```

#### Option 2: Use Your Local Machine
If you have a more powerful local machine, you can:
- Download your repository locally
- Train on the full dataset (may take 1-2 hours)

#### Option 3: Use Google Colab for Training:
   - **Develop locally**: Write and debug your code in Codespaces with small data subsets
   - **Train on Colab with GPU accelleration**: 5-10x faster training compared to CPU. Train on the full dataset (may take 1-2 hours). Remember to save the results to your local machine, as data on Colab is deleted when runtime environment is closed.

### Set up a Python virtual environment and install the required dependencies, if you use a local machine. Note that Codespaces installs all the dependencies in requirements.txt during the initialization, so you can skip this step.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Set Up the Model Upload Client

```bash
# Download the Assignment 4 model upload client
wget https://mse-msci-641.uwaterloo.ca/static/model_upload_client_A4.py

# Initialize the client with your API key
python model_upload_client_A4.py --init
# (Enter the API key provided by the instructor)
```

## Step 3: Implement the Neural Network Classifier

You need to implement and train neural network classifiers with the following specifications:

### Network Architecture
1. **Input layer**: Word2Vec embeddings from Assignment 3
2. **One hidden layer** with different activation functions:
   - ReLU
   - Sigmoid  
   - Tanh
3. **Output layer**: Softmax activation for classification
4. **Loss function**: Cross-entropy loss
5. **Regularization**: L2-norm regularization
6. **Dropout**: Experiment with different dropout rates

### Required Experiments
You must train and evaluate models with:
- **Three different activation functions** in the hidden layer (ReLU, sigmoid, tanh)
- **L2 regularization** - experiment with different values using the validation set
- **Dropout** - experiment with different dropout rates using the validation set
- **Hyperparameter tuning** - use the validation set to select the best hyperparameters
- **Early stopping** - monitor validation performance to prevent overfitting

### Model Submission Requirements
You must submit:
- **Individual models** for each of the three activation functions (ReLU, sigmoid, tanh) with their best hyperparameters found via validation
- **Your best performing model** as the final submission (selected based on validation performance)
- **Experimental results** showing the effect of different hyperparameters on both training and validation performance

**Important:** You may only use the test set for final evaluation of your trained models. All hyperparameter tuning and model selection must be done using the validation set.

## Key Implementation Notes

### Document Embedding Strategy
- **Use averaging, not concatenation**: Average word embeddings to create document-level representations
- **Fixed input size**: All documents must have the same embedding dimension for the neural network
- **Handle OOV words**: Skip words not in your Word2Vec vocabulary
- **Example**: Document with words [w1, w2, w3] → average of [embedding(w1), embedding(w2), embedding(w3)]

### Neural Network Architecture
- **Input size**: Same as Word2Vec embedding dimension (typically 100-300)
- **Hidden layer**: Use powers of 2 for GPU efficiency (64, 128, 256, 512)
  - For input size ~300: try **128** or **256** neurons
  - Powers of 2 are optimized for GPU memory and computation
  - Start with 128, then experiment with 64 or 256 based on results
- **Output layer**: 2 neurons for binary classification (positive/negative sentiment)
- **Activation functions**: Test ReLU, sigmoid, and tanh in the hidden layer

**Why powers of 2?** GPUs are optimized for these sizes, leading to faster training and better memory utilization.

## Implementation Guidelines for `main.py`

The starter code provides function signatures and docstrings for the key components of this assignment. Your task is to complete the implementation of these functions and add the main execution flow.

### 1. DATA LOADING AND PREPROCESSING FUNCTIONS
```python
def load_embeddings(embeddings_path):
    """Load Word2Vec embeddings from Assignment 3"""
    # Your implementation needed here

def load_data_and_labels(data_file, labels_file):
    """Load tokenized data and labels from CSV files"""
    # Your implementation needed here

def create_document_embeddings(tokenized_documents, word2vec_model):
    """Convert tokenized documents to document embeddings using Word2Vec"""
    # Your implementation needed here
```

### 2. NEURAL NETWORK MODEL CLASS
```python
class NeuralNetworkClassifier(nn.Module):
    """Fully-connected neural network classifier"""
    def __init__(self, input_size, hidden_size, num_classes, activation='relu', dropout_rate=0.5):
        # Your implementation needed here
    
    def forward(self, x):
        # Your implementation needed here
```

### 3. TRAINING FUNCTION
```python
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, l2_reg):
    """Train the neural network model"""
    # Your implementation needed here
```

### 4. EVALUATION FUNCTION
```python
def evaluate_model(model, test_loader):
    """Evaluate the model on test data"""
    # Your implementation needed here
```

### 5. MAIN FUNCTION
Implement the main execution flow:
- Load Word2Vec embeddings and data
- Create document embeddings for train/validation/test sets
- For each activation function (ReLU, sigmoid, tanh):
  * Create and train the neural network
  * Experiment with different L2 regularization values
  * Experiment with different dropout rates
  * Evaluate on test data
  * Save the trained model
- Record and save results including activation function, L2 regularization, dropout rate, and accuracy in two files: `hyperparameter_tuning.csv` (all experiments) and `results.csv` (final models) - see "Expected Format for Results" Section for details

## Step 4: Upload Your Trained Models

After training your models, prepare your models directory for upload:

### Prepare Your Models Directory

**Important**: Before uploading, ensure your `models/` directory contains only these 4 files:
- `relu.pth` - Best ReLU activation model
- `sigmoid.pth` - Best sigmoid activation model  
- `tanh.pth` - Best tanh activation model
- `best_model.pth` - Overall best performing model

Remove any other experimental model files to avoid uploading unnecessary models.

### Upload Your Models

**Option 1: Upload all models at once (Recommended)**
```bash
# Upload all models in the models directory
python model_upload_client_a4.py --models_dir models
```

**Option 2: Upload models individually**
```bash
# Upload each activation function model
python model_upload_client_a4.py --model models/relu.pth --config a4_neural_network_relu
python model_upload_client_a4.py --model models/sigmoid.pth --config a4_neural_network_sigmoid  
python model_upload_client_a4.py --model models/tanh.pth --config a4_neural_network_tanh

# Upload your overall best performing model
python model_upload_client_a4.py --best-model models/best_model.pth
```

**What gets uploaded:**
- **3 individual models**: One best model per activation function
- **1 best overall model**: Your highest-performing model
- **Total**: 4 models

### Verify Upload

Check that all models were uploaded successfully:
```bash
# See the list of all uploaded models
python model_upload_client_a4.py --list
```

If you need to delete a previously uploaded model:
```bash
# Delete a specific model (copy the name from the list)
python model_upload_client_a4.py --delete username_a4_neural_network_relu_34690db9
```


## Create a4_model_ids.json file

Run the following command to generate the complete `a4_model_ids.json` file:

```bash
# Generate a4_model_ids.json file containing the names of all your uploaded Assignment 4 models
python model_upload_client_A4.py --generate-json
```

## Step 5: Submit Your Assignment

1. Make sure your repository contains:
   - `main.py` - Your neural network implementation
   - `results.csv` - Your final model results
   - `hyperparameter_tuning.csv` - Your hyperparameter exploration results
   - `a4_model_ids.json` - References to your uploaded models
   - `report.md` - Short report analyzing your results (see below)

2. Commit and push your changes to GitHub:
   ```bash
   git add main.py results.csv hyperparameter_tuning.csv a4_model_ids.json report.md
   git commit -m "Complete Assignment 4"
   git push
   ```

3. The autograder will:
   - Download your models from the server using the IDs in `a4_model_ids.json` 
   - Test your code and models for correctness
   - Verify your results format

## Expected Format for Results

Create **two separate CSV files** to properly document your ML workflow:

### 1. Hyperparameter Tuning Results (`hyperparameter_tuning.csv`)
Document all your experiments during hyperparameter selection:

```
Activation Function,L2 Regularization,Dropout Rate,Validation Accuracy
relu,0.001,0.3,0.851
relu,0.01,0.3,0.845
relu,0.001,0.5,0.843
relu,0.001,0.2,0.849
sigmoid,0.001,0.3,0.825
sigmoid,0.01,0.3,0.822
sigmoid,0.001,0.2,0.828
sigmoid,0.001,0.4,0.820
tanh,0.001,0.3,0.849
tanh,0.01,0.3,0.846
tanh,0.001,0.2,0.852
tanh,0.001,0.4,0.847
```

### 2. Final Results (`results.csv`)
Report your final selected models evaluated on the test set:

```
Activation Function,L2 Regularization,Dropout Rate,Validation Accuracy,Test Accuracy
relu,0.001,0.3,0.851,0.847
sigmoid,0.001,0.2,0.828,0.821
tanh,0.001,0.2,0.852,0.845
```

**Workflow:**
1. **Tune**: Use `hyperparameter_tuning.csv` to document your validation-based hyperparameter search
2. **Select**: Choose the best hyperparameters for each activation function based on validation performance
3. **Evaluate**: Train final models with selected hyperparameters and evaluate once on test set
4. **Report**: Document final results in `results.csv`


## Short Report Requirements

Create a `report.md` file with your analysis (up to 10 sentences). **Use the provided `report_template_a4.md` as your starting point** - simply copy it and fill in your results and analysis.

1. **Effect of activation functions**: What effect do different activation functions (ReLU, sigmoid, tanh) have on your results?
2. **Effect of L2 regularization**: What effect does the addition of L2-norm regularization have on the results?
3. **Effect of dropout**: What effect does dropout have on the results?
4. **Best configuration and Intuitions**: Which configuration gave you the best results and why do you think that is? Explain your intuitions briefly about the observed patterns.

## Grading Criteria

Your assignment will be graded on:
- Correct implementation of the neural network classifier (25%)
- Valid model uploads and proper configurations (15%)
- Experimental design and results format (20%)
- Code quality and organization (15%)
- Short report analysis and insights (15%)
- Best model performance (10%)

## Troubleshooting

### Model Upload Issues

If you have trouble uploading your models:

```bash
# Check if your API key is properly set up
python model_upload_client_A4.py --init

# List your uploaded models
python model_upload_client_A4.py --list

# Try uploading a single model
python model_upload_client_a4.py --model models/relu.pth --config a4_neural_network_relu
```

### PyTorch Installation Issues

If you encounter PyTorch installation issues in Codespaces:

```bash
# Install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Memory Issues

If you run into memory issues:

```bash
# Monitor memory usage
free -h

# Use smaller batch sizes in your training
# Reduce the hidden layer size if necessary
# Use a subset of data for initial testing
```

### Model Size Issues

If your models are too large (>100MB):

- Check your hidden layer size - very large models usually indicate an issue
- Ensure you are saving only the model state dict, not the entire training history

### GitHub Autograding Issues

If autograding fails:
- Check that your `a4_model_ids.json` file is correctly formatted
- Ensure your `results.csv` follows the required format
- Verify that your models were successfully uploaded
- Make sure your main.py can be imported without errors

### Common Problems

1. **Wrong file format**: Make sure your model files are saved as `.pth` or `.pt` files for PyTorch

2. **Missing dependencies**: Ensure PyTorch and other required packages are installed

3. **API key issues**: If you encounter authorization issues, re-initialize your API key

4. **Embedding compatibility**: Make sure your Word2Vec embeddings from Assignment 3 are compatible

5. **CUDA issues**: The server runs on CPU, so ensure your models can run without GPU

### Expected Training Times

- **Simple neural networks**: ~5-10 minutes per model
- **With regularization/dropout**: ~5-15 minutes per model
- **Full experimental suite**: ~1-2 hours total

These times are estimates and may vary based on Codespaces resource availability and your specific implementation.
