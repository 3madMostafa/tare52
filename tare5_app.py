import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re
import torch

# Load the CSV data (Questions and Answers)
df = pd.read_csv('cleaned_output_questions.csv')

# Load a smaller sentence transformer model to reduce resource usage
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # You can switch models based on your use case

# Convert the dataset questions to a list
dataset_questions = df['question'].tolist()

# Compute embeddings for all questions in the dataset
question_embeddings = model.encode(dataset_questions, convert_to_tensor=True)

# Preprocess the question to handle typographical differences and normalization
def preprocess_question(question):
    # Remove unnecessary whitespace
    question = re.sub(r'\s+', ' ', question).strip()
    # Replace common abbreviations or typos (e.g., 'ف' -> 'في')
    question = question.replace(" ف ", " في ")
    return question

# Function to find the most similar question in the dataset using cosine similarity
def find_most_similar_question(new_question, threshold=0.8):
    # Preprocess the input question to handle typographical issues
    new_question = preprocess_question(new_question)
    
    # Compute embedding for the new question
    new_question_embedding = model.encode(new_question, convert_to_tensor=True)

    # Compute cosine similarity between the new question and dataset questions
    cosine_scores = util.pytorch_cos_sim(new_question_embedding, question_embeddings)

    # Find the index of the most similar question (convert the tensor to an integer)
    most_similar_idx = int(cosine_scores.argmax().item())
    most_similar_score = cosine_scores[0, most_similar_idx].item()

    # Print the similarity score for debugging purposes
    st.write(f"Similarity score: {most_similar_score}")

    # Print the most similar question for context
    most_similar_question = dataset_questions[most_similar_idx]
    st.write(f"Most similar question: {most_similar_question}")

    # Check if the similarity score is above the threshold
    if most_similar_score >= threshold:
        # Get the most similar question and its corresponding answer
        most_similar_answer = df['answer'].iloc[most_similar_idx]
        return most_similar_question, most_similar_answer
    else:
        # If the similarity is too low, return a default message
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
