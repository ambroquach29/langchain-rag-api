import streamlit as st
import requests

st.set_page_config(page_title='Langchain RAG')
st.title('RAG Application')
st.write('This is a simple RAG app that allows you to upload a file and ask a question about it.')

# Specify the filename of your local image
# image_filename = 'JacobsSchool.png'
# Use st.image to display the image
# st.image(image_filename, use_container_width=True)

result = None
file_bytes = None

with st.form(key='qa_form'):
    uploaded_file = st.file_uploader('Upload an article', type='txt')
    query_text = st.text_input(
        'Enter your question:', placeholder='Please provide a short summary.')
    google_api_key = st.text_input('Google API Key', type='password')
    submit = st.form_submit_button('Submit')

    if uploaded_file:
        file_bytes = uploaded_file.read()
        # Debug: show first 200 bytes
        st.write("File content preview:", file_bytes[:200])

    if submit:
        if uploaded_file and file_bytes and query_text and google_api_key:
            with st.spinner('Calculating...'):
                files = {'file': (uploaded_file.name, file_bytes)}
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
        else:
            st.warning(
                "Please upload a file, enter your question, and provide your Google API key.")

if result:
    st.info(result)
