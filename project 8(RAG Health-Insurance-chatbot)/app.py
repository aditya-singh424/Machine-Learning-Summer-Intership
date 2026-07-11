import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# --- Page Configuration ---
st.set_page_config(
    page_title="Star Health Insurance RAG Chatbot",
    page_icon="🏥",
    layout="wide"
)

# --- Sidebar Configuration ---
st.sidebar.title("Configuration ⚙️")

# API Key Setup: Check environment variable first, otherwise provide an input field
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# Document path configuration
html_file_path = st.sidebar.text_input(
    "HTML Manual File Path:", 
    value="./starhealth.html",
    help="Ensure this file exists in your workspace directory."
)

# --- Application UI Header ---
st.title("🏥 Star Health Insurance Chatbot")
st.markdown("""
Welcome to the context-aware health insurance support bot. This tool uses Retrieval-Augmented Generation (RAG) 
to answer specific queries based directly on the provided policy manuals.
""")
st.markdown("---")

# --- Core RAG Logic (Cached for Performance) ---
@st.cache_resource(show_spinner="Initializing Vector Database and processing policy documents...")
def initialize_rag_chain(file_path):
    if not os.environ.get("OPENAI_API_KEY"):
        return None
        
    if not os.path.exists(file_path):
        st.error(f"❌ Document file not found at: `{file_path}`. Please verify the file path.")
        return None

    try:
        # 1. Load HTML Content
        loader = UnstructuredHTMLLoader(file_path=file_path)
        machine_docs = loader.load()

        # 2. Text Splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(machine_docs)

        # 3. Embeddings & Vectorstore Setup
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # 4. LLM & Prompt Setup
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = ChatPromptTemplate.from_template(
            "You are an assistant for question-answering tasks.\n"
            "Use the following pieces of retrieved context to answer the question.\n"
            "If you don't know the answer, just say that you don't know.\n"
            "Use three sentences maximum and keep the answer concise.\n\n"
            "Question: {question} \n"
            "Context: {context} \n"
            "Answer:"
        )

        # 5. Build LCEL Chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        return rag_chain
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None

# --- Main Application Logic ---
if not os.environ.get("OPENAI_API_KEY"):
    st.info("🔑 Please enter your OpenAI API key in the sidebar to get started.")
else:
    # Initialize the RAG chain
    rag_chain = initialize_rag_chain(html_file_path)

    if rag_chain:
        # Preset Quick-Sample Queries for User Convenience
        st.subheader("💡 Sample Queries")
        sample_queries = [
            "What are different Maternity Health Insurance Plan in StarHealth?",
            "What benefits do Health Insurance policies offer?",
            "Why should you get a Health Insurance policy when you’re young?",
            "How many types of health insurance policies are there?",
            "What are the different types of health insurance schemes in India?"
        ]
        
        # Display sample queries as clickable buttons
        cols = st.columns(3)
        selected_query = ""
        for i, query_text in enumerate(sample_queries):
            with cols[i % 3]:
                if st.button(query_text, key=f"btn_{i}"):
                    selected_query = query_text

        st.markdown("---")
        st.subheader("💬 Ask Your Own Question")
        
        # Custom input box (prefilled if a sample query button was clicked)
        user_query = st.text_input(
            "Type your insurance query here:", 
            value=selected_query if selected_query else "",
            placeholder="e.g., What are the sub-limits for room rent?"
        )

        if user_query:
            with st.spinner("Searching manuals and generating answer..."):
                try:
                    response = rag_chain.invoke(user_query)
                    
                    # Display Answer
                    st.markdown("### 🤖 Bot Response")
                    st.success(response.content)
                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")