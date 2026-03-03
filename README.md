# FinIntel Engine - ML Models

Machine Learning models and datasets for financial intelligence, including sentiment analysis and technical indicators.

## Project Structure

```
backend/
├── aggregation/
│   └── fundamentalFunctions/     # Fundamental analysis utilities
├── datasets/
│   ├── new_sentiment_dataset/    # Sentiment analysis training data
│   └── technical_datset/         # Technical indicator datasets
├── models/
│   ├── sentiment_model/          # Fine-tuned FinBERT for sentiment
│   └── technical_model/          # GRU-based stock classifier
├── preprocessing/
└── utils/
```

## Getting Started

### Prerequisites

- Git (with Git LFS installed)
- Python 3.8+
- PyTorch, Transformers, TensorFlow/Keras

### Clone the Repository

```bash
# Clone the repo
git clone https://github.com/Kapurrrishabh/FinIntel-Engine.git
cd FinIntel-Engine

# Switch to the ml branch
git checkout ml

# Pull large files (models, datasets)
git lfs pull
```

> **Note:** The repository uses Git LFS for large files (`.safetensors`, `.h5`, `.keras`, `.npy`, `.bin`, `.zip`). Make sure Git LFS is installed before cloning.

### Install Git LFS (if not installed)

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt install git-lfs

# Windows (with Chocolatey)
choco install git-lfs

# Initialize Git LFS
git lfs install
```

### Install Dependencies

```bash
pip install torch transformers datasets scikit-learn matplotlib seaborn tensorflow keras
```

## Usage

### Sentiment Model

The sentiment model is a fine-tuned FinBERT that classifies financial news into Negative, Neutral, or Positive sentiment.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model
model_path = "backend/models/sentiment_model/sentiment_expert_model_v1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Predict sentiment
text = "Apple reports record quarterly revenue, beating analyst expectations"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=-1).item()

labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
print(f"Sentiment: {labels[prediction]}")
```

### Technical Model

```python
from tensorflow.keras.models import load_model
import numpy as np

# Load the model
model = load_model("backend/models/technical_model/gru_stock_classifier-2.keras")

# Load test data
X_test = np.load("backend/datasets/technical_datset/X_test_rnn.npy")
y_test = np.load("backend/datasets/technical_datset/y_test_rnn.npy")

# Predict
predictions = model.predict(X_test)
```

## Contributing

### Create Your Own Branch

```bash
# Make sure you're on the ml branch and up to date
git checkout ml
git pull origin ml

# Create your feature branch
git checkout -b feature/your-feature-name
```

### Making Changes

1. **Add/modify code or models**
2. **Stage your changes:**
   ```bash
   git add .
   ```
3. **Commit with a descriptive message:**
   ```bash
   git commit -m "Add: description of your changes"
   ```

### Working with Large Files

Large files (models, datasets) are automatically tracked by Git LFS. The following extensions are configured:

- `*.safetensors` - Model weights
- `*.h5` - Keras models
- `*.keras` - Keras models
- `*.npy` - NumPy arrays
- `*.bin` - Binary files
- `*.zip` - Archives

To add a new large file type:
```bash
git lfs track "*.your-extension"
git add .gitattributes
```

### Push Your Changes

```bash
# Push your branch to remote
git push -u origin feature/your-feature-name
```

Then create a Pull Request on GitHub to merge into the `ml` branch.

### Sync with Latest Changes

```bash
# Fetch latest changes
git fetch origin

# Merge ml branch into your feature branch
git checkout feature/your-feature-name
git merge origin/ml

# Pull any new LFS files
git lfs pull
```

## Model Details

### Sentiment Model
- **Base Model:** ProsusAI/FinBERT
- **Training Data:** Financial PhraseBank + Twitter Financial News
- **Classes:** Negative (0), Neutral (1), Positive (2)
- **Architecture:** BERT for Sequence Classification

### Technical Model
- **Architecture:** GRU-based classifier
- **Input:** Technical indicators time series
- **Output:** Stock movement prediction

## Troubleshooting

### LFS Files Not Downloading
```bash
# Verify LFS is installed
git lfs version

# Pull LFS files explicitly
git lfs pull

# Or fetch specific files
git lfs fetch --all
```

### Large File Push Rejected
Make sure the file type is tracked by LFS before committing:
```bash
git lfs track "*.your-extension"
git add .gitattributes
git add your-large-file
git commit -m "Add large file with LFS"
```

### Check LFS Tracked Files
```bash
git lfs ls-files
```

## License

This project is for educational purposes.

## Contact

For questions or contributions, please open an issue on GitHub.
