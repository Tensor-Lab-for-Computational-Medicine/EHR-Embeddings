# Google Gemini Embeddings Setup

This project has been updated to use **Google Gemini embeddings** instead of OpenAI embeddings. Gemini embeddings are **free** and provide state-of-the-art performance.

## Benefits of Gemini Embeddings

- **Free to use** with generous quotas
- **State-of-the-art performance** - ranks #1 on MTEB Multilingual leaderboard
- **Higher dimensions** - 3072 dimensions vs OpenAI's 1536-3072
- **Longer context** - 8K tokens vs OpenAI's 8K tokens
- **Multilingual support** - over 100 languages
- **No rate limits** for most usage

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Your Google API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API key"
4. Copy your API key

### 3. Set Environment Variable

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

Or in Python:
```python
import os
os.environ['GOOGLE_API_KEY'] = "your_api_key_here"
```

### 4. Test Your Setup

Run the test cell in the notebook to verify everything is working.

## What Changed

### Original (OpenAI):
- Required paid OpenAI API key
- Used `text-embedding-3-large` model
- Had rate limits and costs

### New (Gemini):
- Uses free Google Gemini API
- Uses `gemini-embedding-exp-03-07` model (latest experimental model)
- Higher performance, no costs for reasonable usage

## Configuration Changes

In your configuration, the following has been updated:

```python
# OLD
'embedding_model': 'text-embedding-3-large'
# Environment variable: OPENAI_API_KEY

# NEW  
'embedding_model': 'gemini-embedding-exp-03-07'
# Environment variable: GOOGLE_API_KEY or GEMINI_API_KEY
```

## Available Models

The implementation supports these Gemini embedding models:

- `gemini-embedding-exp-03-07` (recommended - latest experimental)
- `gemini-embedding-001` (stable)
- `text-embedding-004` (older model)

## Rate Limits

Google Gemini has very generous rate limits:
- 1,500 requests per minute
- 1 million tokens per minute

This is much more generous than OpenAI's limits.

## Troubleshooting

### Error: "Module not found: google.genai"
```bash
pip install google-genai
```

### Error: "API key not found"
Make sure you've set the environment variable:
```bash
export GOOGLE_API_KEY="your_key_here"
```

### Error: "Invalid API key"
- Double-check your API key
- Make sure you're using a valid Google account
- Try regenerating the key at [AI Studio](https://aistudio.google.com/app/apikey)

## Performance Comparison

Based on MTEB benchmarks, Gemini embeddings typically outperform OpenAI embeddings:

- **Gemini embedding**: 68.32 mean score (MTEB Multilingual)
- **OpenAI text-embedding-3-large**: ~64-65 mean score

You should see improved performance with the Gemini embeddings!

## Cost Savings

This change eliminates embedding costs entirely for most use cases:
- **OpenAI**: ~$0.00013 per 1K tokens for text-embedding-3-large
- **Gemini**: Free for up to generous quotas

For a typical Phase 2 analysis with thousands of patient records, this could save hundreds of dollars in API costs. 