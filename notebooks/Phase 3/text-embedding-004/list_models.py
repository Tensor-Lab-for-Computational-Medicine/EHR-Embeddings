# list_models.py
"""
A simple utility script to list all available models for your API key
and check which ones support the 'embedContent' method.
"""
import google.generativeai as genai
import getpass
import os
import logging

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)

def setup_api_key():
    """
    Securely prompts for the user's API key and configures the genai library.
    """
    try:
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            api_key = getpass.getpass('Please enter your Google AI Studio or Vertex AI API key: ')
        genai.configure(api_key=api_key)
        logging.info("Successfully configured API key.")
    except Exception as e:
        logging.error(f"Failed to configure API key: {e}")
        exit(1)

def check_models():
    """
    Lists all available models and their supported generation methods.
    """
    setup_api_key()
    
    print("\n" + "="*60)
    print("      Available Models and Embedding Support")
    print("="*60)
    
    embedding_models_found = []
    
    try:
        for model in genai.list_models():
            # Check if 'embedContent' is one of the supported methods
            if 'embedContent' in model.supported_generation_methods:
                print(f"✅ Model: {model.name}")
                print(f"   - Supports Embedding: Yes")
                embedding_models_found.append(model.name)
            else:
                print(f"❌ Model: {model.name}")
                print(f"   - Supports Embedding: No")
    except Exception as e:
        logging.error(f"Failed to retrieve model list. Error: {e}")
        return

    print("-" * 60)
    if embedding_models_found:
        print("Found the following models suitable for embedding:")
        for name in embedding_models_found:
            print(f"  - {name}")
        print("\nPlease use one of these names in your configuration and test scripts.")
    else:
        print("No models supporting 'embedContent' were found for this API key.")
    print("="*60)


if __name__ == "__main__":
    check_models()