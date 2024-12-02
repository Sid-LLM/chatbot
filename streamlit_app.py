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
Simply select the error number from the dropdown list, and the chatbot will provide relevant information.
""")

errors_list = [
'Over current fault motor',
'Over current fault motor -2',
'Over current fault motor -3',
'Minimum battery voltage fault',
'Over Board temperature fault',
'Battery fault',
'Zigbee fault',
'RTC fault D8',
'Communication fault D11',
'Invalid data',
'Missed rows',
'Invalid distance travelled',
'Position unknown',
'Communication error',
'Cleaner not in position',
'Cleaner error',
'Cleaner stall',
'Track changer stall',
'TC not in position',
'Invalid RFID data Error',
'Invalid Row data Error'
]
error_number = st.selectbox("Choose an error", errors_list, index=0)

custom_error = st.text_input("Or enter a custom error", placeholder="E.g., 707, 808, etc.")
Query = custom_error if custom_error.strip() else error_number

if st.button("Get Details"):
    if Query:
        with st.spinner("Retrieving information..."):
            try:
                # Call the runner method from Inference
                query = Query
                response = rag_inference.runner(query)
                st.subheader("Error's Details and Solutions")
                st.text_area(response, height=300)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid error number.")

# Footer
st.markdown("---")
st.markdown("Built using Streamlit and Hugging Face Transformers.")

