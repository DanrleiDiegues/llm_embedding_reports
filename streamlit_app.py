import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import streamlit as st
import pandas as pd
import google.generativeai as genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure the page
st.set_page_config(
    page_title="AI Agent | Ask About Reports",
    page_icon="🧾🤖",
    layout="centered"
)

# Embedding class (same as you already have)
class GeminiEmbeddingFunction(EmbeddingFunction):
    document_mode = True

    def __call__(self, input: Documents) -> Embeddings:
        embedding_task = "retrieval_document" if self.document_mode else "retrieval_query"
        retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}
        
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=input,
            task_type=embedding_task,
            request_options=retry_policy,
        )
        return response["embedding"]

# Function to load data
@st.cache_resource
def load_knowledge_base():
    data = pd.read_csv("data/reports_2024_texts.csv", 
                      encoding='latin1',
                      sep=',',
                      quotechar='"',
                      escapechar='\\',
                      on_bad_lines='warn')
    return data

# Function to initialize ChromaDB
@st.cache_resource
def init_chromadb():
    embed_fn = GeminiEmbeddingFunction()
    
    # Use the new persistent client
    #chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Use the in-memory client
    chroma_client = chromadb.Client()
    
    db = chroma_client.get_or_create_collection(
        name="reports_db",
        embedding_function=embed_fn
    )
    
    # Load documents if the DB is empty
    if db.count() == 0:
        data = load_knowledge_base()
        documents = data['Content'].tolist()
        db.add(
            documents=documents,
            ids=[str(i) for i in range(len(documents))]
        )
    
    return db, embed_fn

# Initialize the Gemini model
@st.cache_resource
def init_model():
    SYSTEM_MESSAGE = '''You are an agent that answer questions about informations about specific reports from 2024.
                        These are specific informations from different sources, and you are here to show how powerfull is your capability to search knowledge using embeddings from this different sources of knowlege.
                        You will not answer questions that are not related to these reports.
                        If someone asks, you will politely respond that you cannot provide that information.
    '''
    
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-001",
        system_instruction=SYSTEM_MESSAGE
    )
    
    chat = model.start_chat(
        history=[
            {"role": "user", "parts": "Hello"},
            {"role": "model", "parts": "Great to meet you. What would you like to know?"},
        ]
    )
    
    return chat


# Interface principal
def main():
    
    st.title("🧾 Ask About Reports (Single-Agent RAG System)")
    
    # Add examples of questions before the chat input
    st.markdown("""
    This chatbot uses Retrieval-Augmented Generation (RAG) to provide accurate information based on the content of these reports.            
                
    ### Examples of questions you can ask about reports:
    - What are the main topics of the fraud report?
    - What are the main data of Energy report?
    - What are the main climate challenges of 2024 according to the report?
    """)
    
    # Expandable section with examples
    with st.expander("📝 Click here to see examples of questions"):
        st.markdown("""
        ### Topics you can explore:
        
        **Fraud Report:**
        - What are the main topics of the fraud report?
        - What are the main data of Energy report?
        - What are the main climate challenges of 2024 according to the report?
        
        **Energy Report:**
        - What are the main topics of the Energy report?
        - What are the main data of Energy report?
        - What are the main climate challenges of 2024 according to the report?
        
        **Climate Report:**
        - What are the main topics of the Climate report?
        - What are the main data of Climate report?
        - What are the main climate challenges of 2024 according to the report?
        """)
    
    # Initialize components
    db, embed_fn = init_chromadb()
    chat = init_model()
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask your question about reports"):
        # Add question to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Search relevant context
                embed_fn.document_mode = False
                result = db.query(
                    query_texts=[prompt],
                    n_results=2
                )
                context = "\n".join(result["documents"][0])
                
                # Create augmented query
                augmented_query = f"""
                Context: {context}
                
                User question: {prompt}
                
                Please answer the question using the information from the context above.
                """
                
                # Generate response
                response = chat.send_message(augmented_query)
                st.markdown(response.text)
                
                # Add response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.text}
                )

    # Sidebar with information
    with st.sidebar:
        st.markdown("### About")
        st.write("This chatbot uses RAG (Retrieval-Augmented Generation) to provide accurate information about some reports of 2024.")
        
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()
        

if __name__ == "__main__":
    main()