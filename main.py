import os
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv


# 1. Load environment variables
load_dotenv()

# 2. Select provider: 'gemini', 'openai', or 'anthropic'
PROVIDER = os.getenv('PROVIDER', 'gemini').lower()

# 3. Set up API keys and models for each provider
if PROVIDER == 'gemini':
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable not set. Please set it before running.")
    os.environ['GOOGLE_API_KEY'] = api_key
    print("[INFO] Using Gemini provider.")
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")
elif PROVIDER == 'openai':
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. Please set it before running.")
    os.environ['OPENAI_API_KEY'] = api_key
    print("[INFO] Using OpenAI provider.")
    embeddings_model = OpenAIEmbeddings()
    llm = ChatOpenAI()
else:
    raise ValueError(
        f"Unknown PROVIDER: {PROVIDER}. Use 'gemini' or 'openai'.")

# 4. Generate Embeddings for Sample Documents
sample_texts = [
    "This is the Fundamentals of RAG course.",
    "Educative is an AI-powered online learning platform.",
    "There are several Generative AI courses available on Educative.",
    "I am writing this using my keyboard.",
    "JavaScript is a good programming language"
]
print(f"[INFO] Generating embeddings for {len(sample_texts)} sample texts...")
embeddings = embeddings_model.embed_documents(sample_texts)
print(
    f"[INFO] Embeddings generated. Shape: ({len(embeddings)}, {len(embeddings[0])})")

# 5. Prepare Example Documents for Vector Store
documents = [
    "Python is a high-level programming language known for its readability and versatile libraries.",
    "Java is a popular programming language used for building enterprise-scale applications.",
    "JavaScript is essential for web development, enabling interactive web pages.",
    "Machine learning is a subset of artificial intelligence that involves training algorithms to make predictions.",
    "Deep learning, a subset of machine learning, utilizes neural networks to model complex patterns in data.",
    "The Eiffel Tower is a famous landmark in Paris, known for its architectural significance.",
    "The Louvre Museum in Paris is home to thousands of works of art, including the Mona Lisa.",
    "Artificial intelligence includes machine learning techniques that enable computers to learn from data.",
]
print(
    f"[INFO] Preparing {len(documents)} example documents for vector store...")

# 6. Create Chroma Vector Store
print("[INFO] Creating Chroma vector store from documents...")
db = Chroma.from_texts(documents, embeddings_model)
print("[INFO] Chroma vector store created.")

# 7. Configure Retriever
print("[INFO] Configuring retriever (top 1, similarity search)...")
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 1}
)
print("[INFO] Retriever configured.")

# 8. Perform a Similarity Search
query = "Where can I see Mona Lisa?"
print(f"[INFO] Performing similarity search for query: '{query}'")
result = retriever.invoke(query)
print("[RESULT] Retrieved document(s):")
for doc in result:
    print(f"- {doc.page_content if hasattr(doc, 'page_content') else doc}")
print('\n')

# 9. Define a template for generating answers using provided context
template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say 'thanks for asking!' at the end of the answer.

{context}
Question: {question}

Helpful Answer:
"""

# 10. Create a custom prompt template using the defined template
custom_rag_prompt = PromptTemplate.from_template(template)
print('Custom RAG Prompt:--------------------------------')
print(custom_rag_prompt)  # Print the custom prompt template
print()

# Assume retriever is already defined and configured
question = "What is the future of AI?"
# Retrieve the context based on the question
context = retriever.invoke(question)
print('Context:--------------------------------')
print(context)
print()

# Manually format the prompt template to see the augmented query
augmented_query = custom_rag_prompt.format(context=context, question=question)
print("Augmented Query:--------------------------------")
print(augmented_query)
print()

# 1. Add a function to format the retrieved documents


def format_docs(docs):
    """Joins the page_content of retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# 2. LLM is already initialized above


# 3. Construct the corrected chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

# 4. Invoke the chain
print("[INFO] Invoking RAG chain...")
response = rag_chain.invoke("What is the future of AI?")
print("\n[Final Answer]")
print(response)
