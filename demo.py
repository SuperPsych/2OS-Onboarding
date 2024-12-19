import openai
from datasets import load_dataset
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st

openai.api_key = ""

def get_gpt_score(sentence1, sentence2, model="gpt-4o-mini"):
    prompt = f"Your job is to provide a semantic similarity score between Sentence 1 and Sentence 2. DO NOT SAY ANY WORDS, PROVIDE A SINGLE REAL NUMBER BETWEEN 0 AND 5.\n" \
             f"Sentence 1: {sentence1}\n" \
             f"Sentence 2: {sentence2}\n" \
             f"Score: "
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Your only job is to output numbers, you do not say words."},
                {"role": "user", "content": prompt}
            ]
        )
        score = float(response.choices[0].message.content.strip())
        return score
    except Exception as e:
        print(f"Error querying model: {e}")
        return None


def get_cosine_similarity(sentence1, sentence2, model="text-embedding-ada-002"):
    try:
        embedding1 = openai.Embedding.create(input=sentence1, model=model)["data"][0]["embedding"]
        embedding2 = openai.Embedding.create(input=sentence2, model=model)["data"][0]["embedding"]
        return compute_cosine_similarity(embedding1, embedding2) * 5
    except Exception as e:
        print(f"Error querying embeddings model: {e}")
        return None


def compute_cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


dataset = load_dataset("mteb/stsbenchmark-sts")
data = dataset["train"].select(range(10))

sample_df = pd.DataFrame(data)

sentence1_list = sample_df["sentence1"].tolist()
sentence2_list = sample_df["sentence2"].tolist()
true_scores = sample_df["score"].tolist()

sample_df["GPT-4o-mini Score"] = [get_gpt_score(s1, s2, model="gpt-4o-mini") for s1, s2 in
                                  zip(sentence1_list, sentence2_list)]

sample_df["ChatGPT Embeddings Score"] = [
    get_cosine_similarity(s1, s2, model="text-embedding-ada-002") for s1, s2 in zip(sentence1_list, sentence2_list)
]

gpt_rmse = np.sqrt(mean_squared_error(true_scores, sample_df["GPT-4o-mini Score"].fillna(0)))
cosine_rmse = np.sqrt(mean_squared_error(true_scores, sample_df["ChatGPT Embeddings Score"].fillna(0)))

gpt_mae = mean_absolute_error(true_scores, sample_df["GPT-4o-mini Score"].fillna(0))
cosine_mae = mean_absolute_error(true_scores, sample_df["ChatGPT Embeddings Score"].fillna(0))

st.title("LLM Model Evaluation Dashboard")

st.header("Sample Dataset")
st.write(sample_df)

st.header("Model Performance Metrics")
metrics = pd.DataFrame({
    "Model": ["GPT-4o-mini", "ChatGPT Embeddings"],
    "RMSE": [gpt_rmse, cosine_rmse],
    "MAE": [gpt_mae, cosine_mae]
})
st.table(metrics)


#GPT-4o-mini almost always performs better here.
#I believe this is because the embeddings model overestimates the semantic similarity of two sentences
# that are similar in structure, but very different in meaning.
st.header("Conclusion")
if gpt_rmse < cosine_rmse:
    st.write("According to the RMSE metric, GPT-4o-mini performed better than ChatGPT Embeddings!")
else:
    st.write("According to the RMSE metric, ChatGPT Embeddings performed better than GPT-4o-mini!")

def get_summary(article, model):
    prompt = f"Summarize the following article in one concise sentence:\n{article}\nSummary:"
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful summarization assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"Error querying model for summarization: {e}")
        return ""

def compute_rouge(reference, generated):
    if not isinstance(reference, str):
        reference = str(reference)
    if not isinstance(generated, str):
        generated = str(generated)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {key: value.fmeasure for key, value in scores.items()}

dataset = load_dataset("xsum", trust_remote_code=True)
data = dataset["train"].select(range(10))

sample_df = pd.DataFrame(data)

articles = sample_df["document"].tolist()
true_summaries = sample_df["summary"].tolist()

sample_df["GPT-4o-mini Summary"] = [get_summary(article, model="gpt-4o-mini") for article in articles]

sample_df["GPT-3.5-turbo Summary"] = [get_summary(article, model="gpt-3.5-turbo") for article in articles]

sample_df["ROUGE-1 (GPT-4o-mini)"] = [
    compute_rouge(ref, gen).get("rouge1", 0) for ref, gen in zip(true_summaries, sample_df["GPT-4o-mini Summary"])
]

sample_df["ROUGE-1 (GPT-3.5-turbo)"] = [
    compute_rouge(ref, gen).get("rouge1", 0) for ref, gen in zip(true_summaries, sample_df["GPT-3.5-turbo Summary"])
]

st.header("Sample Dataset")
st.write(sample_df[["document", "summary", "GPT-4o-mini Summary", "GPT-3.5-turbo Summary"]])

rogue_4o = np.mean(sample_df["ROUGE-1 (GPT-4o-mini)"].fillna(0))
rogue_3_5 = np.mean(sample_df["ROUGE-1 (GPT-3.5-turbo)"].fillna(0))
rouge_metrics = pd.DataFrame({
    "Model": ["GPT-4o-mini", "GPT-3.5-turbo"],
    "Average ROUGE-1": [
        rogue_4o,
        rogue_3_5
    ]
})

st.header("ROUGE Metrics for Summarization")
st.table(rouge_metrics)

st.header("Conclusion")
if rogue_4o < rogue_3_5:
    st.write("According to the RMSE metric, GPT-3.5 performed better than GPT-4o-mini!")
else:
    st.write("According to the RMSE metric, GPT-4o-mini performed better than GPT-3.5!")
