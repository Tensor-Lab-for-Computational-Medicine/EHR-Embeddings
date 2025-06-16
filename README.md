# EHR Embeddings with Google Gemini API

A comprehensive Python environment for utilizing Google's Gemini embedding API to encode Electronic Health Record (EHR) data and build machine learning models for clinical predictions.

## 🚀 Features

- **Google Gemini Integration**: Seamless integration with Google's Gemini embedding API
- **EHR Text Processing**: Specialized preprocessing for medical text data
- **Embedding Caching**: Intelligent caching system to reduce API calls and costs
- **ML Pipeline**: Complete machine learning pipeline with multiple algorithms
- **Model Evaluation**: Comprehensive model evaluation and comparison tools
- **Hyperparameter Tuning**: Automated hyperparameter optimization
- **Jupyter Notebooks**: Interactive examples and tutorials

## 📋 Requirements

- Python 3.8+
- Google API key for Gemini
- Required packages (see `requirements.txt`)

## 🛠️ Installation

### Quick Setup

1. **Clone or download the project**
2. **Run the setup script**:
   ```bash
   python setup.py
   ```

### Manual Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Create directories**:
   ```bash
   mkdir -p data embeddings_cache models logs
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your Google API key
   ```

## 🔑 API Key Setup

1. Get your Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add it to your `.env` file:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

## 📖 Usage

### Basic Example

```python
from src.embeddings import GeminiEmbedder
from src.ml_pipeline import EHRMLPipeline, create_sample_ehr_data

# Initialize embedder
embedder = GeminiEmbedder()

# Create sample EHR data
data = create_sample_ehr_data(n_samples=100)

# Generate embeddings
text_data = data['chief_complaint'] + " | " + data['diagnosis']
embeddings = embedder.embed_batch(text_data.tolist())

# Train ML models
ml_pipeline = EHRMLPipeline(task_type="classification")
X_train, X_test, y_train, y_test = ml_pipeline.prepare_data(
    df=data, target_column='readmission_risk'
)
results = ml_pipeline.train_models(X_train, y_train)
```

### Run Complete Example

```bash
python example_usage.py
```

### Jupyter Notebook

```bash
jupyter notebook notebook_example.ipynb
```

## 📁 Project Structure

```
ehr_embeddings/
├── src/                   # Source code
│   ├── __init__.py
│   ├── embeddings.py      # Gemini API integration
│   ├── ml_pipeline.py     # ML pipeline and models
│   └── README.md          # Source documentation
├── data/                  # Data storage
│   ├── raw/               # Raw, unprocessed data
│   │   ├── all_hourly_data.h5  # Main EHR dataset (7.2GB)
│   │   └── mimic_legacy.txt    # Legacy MIMIC data
│   ├── processed/         # Processed data files
│   └── README.md          # Data documentation
├── notebooks/             # Jupyter notebooks
│   ├── h5_first_5_rows.ipynb   # Data exploration
│   ├── example_usage.ipynb     # Usage examples
│   └── README.md               # Notebook documentation
├── outputs/               # Generated outputs
│   ├── figures/           # Plots and visualizations
│   ├── models/            # Trained ML models
│   ├── reports/           # Analysis reports
│   └── README.md          # Output documentation
├── docs/                  # Project documentation
│   └── EHR Embeddings Analysis Plan.md
├── cache/                 # Cached embeddings
│   ├── *.pkl              # Cached embedding files
│   └── README.md          # Cache documentation
├── logs/                  # Log files
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── setup.py              # Setup script
├── example_usage.py      # Complete usage example
├── README.md             # This file
├── .gitignore            # Git ignore rules
└── __pycache__/          # Python cache files
```

## 🧩 Core Components

### 1. GeminiEmbedder (`src/embeddings.py`)

Main class for generating embeddings using Google's Gemini API:

- **Text preprocessing** for medical data
- **Batch processing** with rate limiting
- **Intelligent caching** to reduce API costs
- **Error handling** with exponential backoff

Key methods:
- `embed_text(text)`: Generate embedding for single text
- `embed_batch(texts)`: Process multiple texts efficiently
- `embed_ehr_data(dataframe)`: Process entire EHR datasets

### 2. EHRMLPipeline (`src/ml_pipeline.py`)

Complete machine learning pipeline:

- **Multiple algorithms**: Random Forest, XGBoost, LightGBM, SVM, Logistic Regression
- **Automated preprocessing**: Feature scaling, encoding
- **Cross-validation**: Robust model evaluation
- **Hyperparameter tuning**: Grid search optimization
- **Model persistence**: Save/load trained models

Key methods:
- `prepare_data()`: Prepare embeddings for ML
- `train_models()`: Train multiple models with CV
- `evaluate_model()`: Comprehensive model evaluation
- `hyperparameter_tuning()`: Automated parameter optimization

## 🎯 Use Cases

### Clinical Prediction Tasks

1. **Readmission Risk**: Predict patient readmission probability
2. **Diagnosis Classification**: Classify conditions from clinical notes
3. **Length of Stay**: Predict hospital stay duration
4. **Adverse Events**: Identify risk of complications
5. **Treatment Response**: Predict treatment effectiveness

### Supported Data Types

- **Clinical Notes**: Doctor's notes, discharge summaries
- **Chief Complaints**: Patient-reported symptoms
- **Diagnosis Codes**: ICD-10 codes with descriptions
- **Medication Lists**: Drug names and dosages
- **Lab Results**: Test results with interpretations

## ⚙️ Configuration

Configure the system via environment variables in `.env`:

```bash
# API Configuration
GOOGLE_API_KEY=your_api_key
EMBEDDING_MODEL=models/embedding-001

# Processing
BATCH_SIZE=100
MAX_RETRIES=3

# Storage
EHR_DATA_PATH=data/ehr_data.csv
EMBEDDINGS_CACHE_DIR=embeddings_cache/
MODEL_OUTPUT_DIR=models/
```

## 📊 Model Performance

The pipeline supports various evaluation metrics:

### Classification Tasks
- Accuracy, Precision, Recall, F1-score
- ROC curves and AUC scores
- Confusion matrices
- Classification reports

### Regression Tasks
- MSE, MAE, RMSE
- R² scores
- Residual analysis

## 🔧 Advanced Features

### Embedding Caching
Automatic caching reduces API costs:
```python
# Embeddings are automatically cached
embedder = GeminiEmbedder()
embeddings = embedder.embed_batch(texts, use_cache=True)
```

### Hyperparameter Tuning
Automated optimization for better performance:
```python
tuning_results = ml_pipeline.hyperparameter_tuning(
    X_train, y_train, 
    model_name="random_forest"
)
```

### Feature Importance
Understand model decisions:
```python
importance_df = ml_pipeline.get_feature_importance()
```

## 🔒 Privacy & Security

### HIPAA Compliance Considerations

- **Data Anonymization**: Remove PHI before embedding
- **Local Processing**: Embeddings can be cached locally
- **API Security**: Use secure API key management
- **Audit Logging**: Track all data processing

### Best Practices

1. **Remove PHI** before generating embeddings
2. **Use environment variables** for API keys
3. **Implement access controls** for sensitive data
4. **Monitor API usage** and costs
5. **Regular security audits** of the pipeline

## 📈 Performance Optimization

### Embedding Generation
- Use batch processing for efficiency
- Implement caching to avoid regenerating embeddings
- Monitor API rate limits

### Model Training
- Use cross-validation for robust evaluation
- Implement early stopping for deep learning models
- Consider ensemble methods for better performance

### Memory Management
- Process large datasets in chunks
- Use efficient data structures (numpy arrays)
- Clear memory after processing batches

## 🐛 Troubleshooting

### Common Issues

1. **API Key Not Found**:
   ```
   ValueError: Google API key is required
   ```
   Solution: Set `GOOGLE_API_KEY` in your `.env` file

2. **Rate Limiting**:
   ```
   Error: API rate limit exceeded
   ```
   Solution: Reduce `BATCH_SIZE` in configuration

3. **Memory Issues**:
   ```
   MemoryError: Unable to allocate array
   ```
   Solution: Process data in smaller batches

4. **Import Errors**:
   ```
   ModuleNotFoundError: No module named 'src'
   ```
   Solution: Run from project root directory

### Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📜 License

This project is provided as-is for educational and research purposes. Please ensure compliance with all applicable regulations when handling medical data.

## 🔗 Resources

- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [HIPAA Guidelines](https://www.hhs.gov/hipaa/index.html)
- [Medical NLP Best Practices](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6371183/)

## 💬 Support

For questions and support:
1. Check the troubleshooting section
2. Review the example notebook
3. Run the example script to verify setup
4. Check configuration in `.env` file

---

**Note**: This tool processes medical data. Always ensure compliance with relevant privacy regulations and institutional policies before using with real patient data. # EHR-Embeddings
