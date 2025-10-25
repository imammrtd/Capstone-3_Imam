# import os
import streamlit as st
import pandas as pd

QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage
from langchain_core.documents import Document

from dotenv import load_dotenv



llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key = OPENAI_API_KEY
)

# Setup Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# Setup Qdrant dengan error handling
collection_name = "imdb_movies"

class document:
    def __init__(self, page_content, metadata, id=None):
        self.page_content = page_content
        self.metadata = metadata
        self.id = id or str(hash(page_content))




try:
    print("Trying to access existing collection...")
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    print("Collection accessed successfully!")
except Exception as e:
    print(f"Error accessing collection: {e}")
    st.warning("Collection not found. Uploading data from CSV...")
    try:
        df = pd.read_csv('imdb_top_1000.csv')  # Load df di sini
        print("CSV loaded successfully!")
    except FileNotFoundError:
        st.error("File 'imdb_top_1000.csv' not found. Please download the IMDb dataset.")
        st.stop()
    
    # Inisialisasi documents
    df = pd.read_csv('imdb_top_1000.csv')
    documents = []
    for _, row in df.iterrows():
        content = f"Title: {row['Series_Title']}, Year: {row['Released_Year']}, Genre: {row['Genre']}, Rating: {row['IMDB_Rating']}, Overview: {row['Overview']}"
        documents.append(document(page_content=content, metadata={"title": row['Series_Title']}))  # Perbaiki typo jadi Document
    
    print(f"Uploading {len(documents)} documents...")

    # Upload ke Qdrant
    qdrant = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=QDRANT_URL,
    )
    st.success("Data uploaded successfully!")

 
collection_name = "imdb_movies"
qdrant = QdrantVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=collection_name
)
print("Upload berhasil!")
  

@tool
def get_relevant_docs(question):
  """Use this tool to get relevant movie documents from the IMDb dataset."""
  results = qdrant.similarity_search(
      question,
      k=5
  )
  return results

tools = [get_relevant_docs]

def chat_movie(question, history):
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt= f'''You are an expert on movies. Answer only questions about movies and use given tools to retrieve movie details from the IMDb dataset.'''
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    answer = result["messages"][-1].content

    total_input_tokens = 0
    total_output_tokens = 0

    for message in result["messages"]:
        if "usage_metadata" in message.response_metadata:
            total_input_tokens += message.response_metadata["usage_metadata"]["input_tokens"]
            total_output_tokens += message.response_metadata["usage_metadata"]["output_tokens"]
        elif "token_usage" in message.response_metadata:
            # Fallback for older or different structures
            total_input_tokens += message.response_metadata["token_usage"].get("prompt_tokens", 0)
            total_output_tokens += message.response_metadata["token_usage"].get("completion_tokens", 0)

    price = 17_000*(total_input_tokens*0.15 + total_output_tokens*0.6)/1_000_000

    tool_messages = []
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            tool_message_content = message.content
            tool_messages.append(tool_message_content)

    response = {
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages
    }
    return response

st.title("Chatbot Movie Master")
st.image("./Movie Master Agent/header_img.png")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me movie questions"):  # Ganti prompt
    messages_history = st.session_state.get("messages", [])[-20:]
    history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history]) or " "

    # Display user message in chat message container
    with st.chat_message("Human"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "Human", "content": prompt})
    
    # Display assistant response in chat message container
    with st.chat_message("AI"):
        response = chat_movie(prompt, history)  # Ganti nama fungsi
        answer = response["answer"]
        st.markdown(answer)
        st.session_state.messages.append({"role": "AI", "content": answer})

    with st.expander("**Movie Tool Calls:**"):  # Ganti judul
        st.code(response["tool_messages"])

    with st.expander("**Movie Chat History:**"):  # Ganti judul
        st.code(history)

    with st.expander("**Token Usage Details:**"):  # Ganti judul
        st.code(f'input token : {response["total_input_tokens"]}\noutput token : {response["total_output_tokens"]}')