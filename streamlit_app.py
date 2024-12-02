# app.py

import streamlit as st
from RAG import Inference

# Initialize the inference object
rag_inference = Inference()

# Streamlit app
st.set_page_config(page_title="Error Resolution Chatbot", layout="centered")

# App Title
st.title("Error Resolution Chatbot")
st.markdown("""
This app helps you retrieve solutions and details about errors in your product.
Simply enter the error number, and the chatbot will provide relevant information.
""")

# Input Query
st.subheader("Enter the Error Number")
error_number = st.text_input("Error Number", placeholder="E.g., 101, 202, etc.")

if st.button("Get Details"):
    if error_number.strip():
        with st.spinner("Retrieving information..."):
            try:
                # Call the runner method from Inference
                query = f"Provide me information about the error number: {error_number}"
                response = rag_inference.runner(query)
                st.subheader("Results")
                st.text_area("Response", response, height=300)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid error number.")

# Footer
st.markdown("---")
st.markdown("Built using Streamlit and Hugging Face Transformers.")

