# Import semua library
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage
from langchain_core.documents import Document
import pandas as pd
import matplotlib.pyplot as plt

# Secrets
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Setup LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY
)

# Setup Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# Setup Qdrant dengan error handling
collection_name = "imdb_movies"
try:
    st.write("Trying to access existing Qdrant collection...")
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    st.success("Collection accessed successfully!")

except Exception as e:
    st.warning(f"Error accessing collection: {e}")
    st.info("Collection not found. uploading data from CSV...")
    try:
        df = pd.read_csv('imdb_top_1000.csv')  
        print("CSV loaded successfully!")
    except FileNotFoundError:
        st.error("File 'imdb_top_1000.csv' not found. Please download the IMDb dataset.")
        st.stop()
    
    # Inisialisasi documents
    documents = []
    for _, row in df.iterrows():
        content = f"Title: {row['Series_Title']}, Year: {row['Released_Year']}, Genre: {row['Genre']}, Rating: {row['IMDB_Rating']}, Overview: {row['Overview']}"
        documents.append(Document(page_content=content, metadata={"title": row['Series_Title']})) 
    
    print(f"Uploading {len(documents)} documents...")

    # Upload ke Qdrant
    qdrant = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name
    )
    st.success("Data uploaded successfully!")
    print("Upload completed!")

# Tool
@tool
def get_relevant_docs(question):
    """Use this tool to get relevant movie documents from the IMDb dataset."""
    results = qdrant.similarity_search(question, k=5)
    return results

tools = [get_relevant_docs]

# Function chat_movie
def chat_movie(question, history):
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt="""You are an expert on movies. Answer only questions about movies and use given tools to retrieve movie details from the IMDb dataset."""
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].content

    # Token calculation
    total_input_tokens = 0
    total_output_tokens = 0
    for message in result["messages"]:
        if "usage_metadata" in message.response_metadata:
            total_input_tokens += message.response_metadata["usage_metadata"]["input_tokens"]
            total_output_tokens += message.response_metadata["usage_metadata"]["output_tokens"]
        elif "token_usage" in message.response_metadata:
            total_input_tokens += message.response_metadata["token_usage"].get("prompt_tokens", 0)
            total_output_tokens += message.response_metadata["token_usage"].get("completion_tokens", 0)

    price = 17_000 * (total_input_tokens * 0.15 + total_output_tokens * 0.6) / 1_000_000

    tool_messages = []
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            tool_messages.append(message.content)

    response = {
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages
    }
    return response

# UI
st.title("Chatbot Movie Master")
header_path = "./Movie Master Agent/header_img.png"
if os.path.exists(header_path):
    st.image(header_path)
else:
    st.warning("Header image not found.")

# Movie Analytics Dashboard
tab1, tab2 = st.tabs(["Chat", "Analytics"])

with tab2:
    st.header("Movie Analytics Dashboard")
    st.markdown("Average IMDB Rating per Genre (Top 10)")
    try:
        avg_rating = df.groupby("Genre")["IMDB_Rating"].mean().sort_values(ascending=False).head(10)
        plt.figure(figsize=(8, 4))
        plt.barh(avg_rating.index, avg_rating.values)
        plt.xlabel("Average Rating")
        plt.ylabel("Genre")
        plt.gca().invert_yaxis()
        st.pyplot(plt)
    except Exception as e:
        st.write(f"⚠️ Error loading analytics: {e}")


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask me movie questions"):
    # Tone Detection
    sentiment_prompt = f"Analyze the tone of this message: '{prompt}'. Reply with one word only: Positive, Neutral, or Confused."
    try:
        tone = llm.invoke(sentiment_prompt).content
    except Exception:
        tone = "Unknown"
    st.caption(f"🗣️ User tone detected: {tone}")

    messages_history = st.session_state.get("messages", [])[-20:]
    history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history]) or " "

    with st.chat_message("Human"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "Human", "content": prompt})
    
    with st.chat_message("AI"):
        response = chat_movie(prompt, history)
        answer = response["answer"]
        st.markdown(answer)
        st.session_state.messages.append({"role": "AI", "content": answer})
    
    # Recommended Similar Movies
        st.subheader("Recommended Similar Movies")
        main_keyword = prompt.split()[0]
        try:
            similar_movies = qdrant.similarity_search(main_keyword, k=5)
            for doc in similar_movies:
                st.write(f"- {doc.page_content}")
        except Exception as e:
            st.warning(f"⚠️ Could not fetch recommendations: {e}")

    with st.expander("**Movie Tool Calls:**"):
        st.code(response["tool_messages"])

    with st.expander("**Movie Chat History:**"):
        st.code(history)

    with st.expander("**Token Usage Details:**"):
        st.code(f'input token : {response["total_input_tokens"]}\noutput token : {response["total_output_tokens"]}')
