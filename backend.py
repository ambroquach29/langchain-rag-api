from langchain import hub
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def generate_response(file_content, google_api_key, query_text):
    documents = [file_content]
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=500, chunk_overlap=100)
    texts = text_splitter.create_documents(documents)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite-preview-06-17", google_api_key=google_api_key)
    # Select embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=google_api_key)
    # Create a vectorstore from documents
    database = Chroma.from_documents(texts, embeddings)
    # Create retriever interface
    retriever = database.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # Create QA chain
    response = rag_chain.invoke(query_text)
    return response
