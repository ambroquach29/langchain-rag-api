import streamlit as st
import requests

st.set_page_config(page_title='Langchain RAG')
st.title('RAG Application')
st.write('This is a simple RAG app that allows you to upload a file and ask a question about it.')

# Specify the filename of your local image
image_filename = 'JacobsSchool.png'
# Use st.image to display the image
st.image(image_filename, use_container_width=True)

# File upload
uploaded_file = st.file_uploader('Upload an article', type='txt')
# Query text
query_text = st.text_input(
    'Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_file)
google_api_key = st.text_input(
    'Google API Key', type='password', disabled=not (uploaded_file and query_text))

result = None

if uploaded_file and query_text and google_api_key.startswith('sk-'):
    if st.button('Submit'):
        with st.spinner('Calculating...'):
            files = {'file': uploaded_file}
            data = {
                'google_api_key': google_api_key,
                'query_text': query_text
            }
            try:
                response = requests.post(
                    "http://localhost:8000/generate-response",
                    files=files,
                    data=data,
                    timeout=120
                )
                response.raise_for_status()
                result = response.json().get("result", "No result returned.")
            except Exception as e:
                result = f"Error: {e}"

if result:
    st.info(result)
