import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re
import torch

# Load the CSV data (Questions and Answers)
df = pd.read_csv('cleaned_output_questions (1).csv')

# Load a more powerful multilingual sentence transformer model
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Ensure the model runs on GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Preprocess the question to handle typographical differences and normalization
def preprocess_question(question):
    question = re.sub(r'\s+', ' ', question).strip()
    question = question.replace(" ف ", " في ")
    return question

# Function to find the most similar question in the dataset using cosine similarity
def find_most_similar_question(new_question, threshold=0.8):
    new_question = preprocess_question(new_question)
    new_question_embedding = model.encode(new_question, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(new_question_embedding, question_embeddings)
    most_similar_idx = int(cosine_scores.argmax().item())
    most_similar_score = cosine_scores[0, most_similar_idx].item()
    most_similar_question = dataset_questions[most_similar_idx]
    
    if most_similar_score >= threshold:
        most_similar_answer = df['answer'].iloc[most_similar_idx]
        return most_similar_question, most_similar_answer
    else:
        return None, "I don't know the answer to that question."

# Main function to generate an answer
def generate_hybrid_answer(question):
    similar_question, answer = find_most_similar_question(question)
    return answer

# Streamlit Web Application
st.title("Arabic Question Answering System")
st.write("Ask a question, and get the best-matching answer from our dataset.")

# Input for the user's question
user_question = st.text_input("Enter your question:")

# Display answer when the user inputs a question
if user_question:
    answer = generate_hybrid_answer(user_question)
    st.write(f"Answer: {answer}")
