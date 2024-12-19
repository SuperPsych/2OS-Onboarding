# README

## Overview
This project evaluates and compares the performance of various large language models (LLMs) in two tasks:
1. **Semantic Similarity**: Comparing GPT-4o-mini and ChatGPT Embeddings models using a semantic similarity dataset.
2. **Summarization**: Comparing GPT-4o-mini and GPT-3.5-turbo models using the XSum summarization dataset, evaluated with ROUGE metrics.

## Libraries Used
1. **`openai`**: To interact with GPT models for both semantic similarity and summarization tasks.
2. **`datasets`**: To load Hugging Face datasets (`mteb/stsbenchmark-sts` and `xsum`).
3. **`pandas`**: For data manipulation.
4. **`numpy`**: For numerical computations.
5. **`rouge_score`**: To compute ROUGE metrics for summarization performance evaluation.
6. **`sklearn.metrics`**: For RMSE and MAE calculations.
7. **`streamlit`**: To create an interactive dashboard for visualizing the results.

## Flow of Information
### **Semantic Similarity Task**
1. **Dataset**: Uses the `mteb/stsbenchmark-sts` dataset (10 examples).
2. **Models**:
   - GPT-4o-mini: Generates similarity scores via prompts.
   - ChatGPT Embeddings: Computes cosine similarity scores from embeddings.
3. **Evaluation**:
   - RMSE and MAE between model predictions and the ground-truth scores.
   - Visualized via a Streamlit dashboard.

### **Summarization Task**
1. **Dataset**: Uses the `xsum` dataset (10 examples).
2. **Models**:
   - GPT-4o-mini: Generates summaries via prompts.
   - GPT-3.5-turbo: Generates summaries via prompts.
3. **Evaluation**:
   - ROUGE-1 scores for both models compared to ground-truth summaries.
   - Results displayed in the Streamlit dashboard.

## Setup Instructions
### 1. Clone the Repository
```bash
# Clone this repository
git clone <repository-url>
cd <repository-directory>
```

### 2. Install Required Libraries
Ensure you have Python 3.8+ and install the dependencies:
```bash
pip install openai datasets pandas numpy rouge_score scikit-learn streamlit
```

### 3. Set Up OpenAI API Key
Replace the `openai.api_key` value with your OpenAI API key in the code.

### 4. Run the Streamlit Dashboard
Run the application locally using Streamlit:
```bash
streamlit run demo.py
```

*might take some time to download the dataset on the first run

Step 1: 10 minutes

Step 2: 2 hours

Step 3/4: 3 hours

Step 5: 30 minutes