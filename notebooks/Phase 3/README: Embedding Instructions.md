# Instructions for Parallel Embedding Generation

This guide explains how to split the work of generating ~400,000 embeddings across three people. Each person will be responsible for running the script for their assigned "worker ID".

### Prerequisites

1.  **Python Environment**: Ensure you have Python installed.
2.  **Required Libraries**: Install the necessary libraries by running:
    ```bash
    pip install -q google-generativeai numpy tqdm
    ```
3.  **API Key**: Have your personal Google AI Studio API key ready.
4.  **Project Files**: Make sure you have the following files in the same directory:
    * `generate_embeddings.py` (the main script)
    * `config_embedding.py` (the configuration file)

### Step 1: Assign Worker IDs

Decide among the three of you who will be Worker 0, Worker 1, and Worker 2.

* **Person 1**: Worker ID `0`
* **Person 2**: Worker ID `1`
* **Person 3**: Worker ID `2`

### Step 2: Perform a Dry Run (Highly Recommended)

Before starting the full process, each person should perform a quick "dry run" to ensure their setup and API key are working correctly.

1.  **Open `config_embedding.py`** and make sure `DRY_RUN` is set to `True`.
    ```python
    DRY_RUN = True
    ```
2.  **Open your terminal** or command prompt, navigate to the directory containing the scripts.
3.  **Run the script with your assigned Worker ID.**
    * **Person 1 (Worker 0) runs:**
        ```bash
        python generate_embeddings.py --worker-id 0
        ```
    * **Person 2 (Worker 1) runs:**
        ```bash
        python generate_embeddings.py --worker-id 1
        ```
    * **Person 3 (Worker 2) runs:**
        ```bash
        python generate_embeddings.py --worker-id 2
        ```
4.  **Enter your API Key** when prompted.
5.  **Verify**: The script should process one file and exit. Check the `notebooks/Phase 4/phase_4_embeddings` directory to confirm that a single `.npy` file was created.

### Step 3: Run the Full Process

Once everyone has successfully completed a dry run, you are ready to process all the files.

1.  **Open `config_embedding.py`** and change `DRY_RUN` to `False`.
    ```python
    DRY_RUN = False
    ```
2.  **Save the file.**
3.  **Run the script again** with your assigned worker ID, just like in the dry run.
    * **Person 1 (Worker 0) runs:**
        ```bash
        python "notebooks/Phase 3/generate_embeddings.py" --worker-id 0
        ```
    * **Person 2 (Worker 1) runs:**
        ```bash
        python "notebooks/Phase 3/generate_embeddings.py" --worker-id 1
        ```
    * **Person 3 (Worker 2) runs:**
        ```bash
        python "notebooks/Phase 3/generate_embeddings.py" --worker-id 2
        ```
4.  The script will now start processing its assigned third of the files. A progress bar will show you how long it will take. You can let it run in the background.

By following these steps, your team can efficiently and safely generate all the embeddings without duplicating work.
