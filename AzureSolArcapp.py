
import streamlit as st
import os
from groq import Groq

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from tqdm.auto import tqdm # Used for loading chunks initially, though not directly in the Streamlit app for *new* processing

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Azure AI Q&A Chatbot", layout="centered")
st.title("Azure AI Services & OpenAI Q&A")
st.write("Ask questions about the Azure AI Services & OpenAI document.")

# --- API Key Input ---
groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
# Although the current embedding model 'thenlper/gte-large' typically doesn't require
# a Hugging Face API key for local inference, a placeholder is included if you use other models.
hf_api_key = st.sidebar.text_input("Enter your Hugging Face API Key (optional for this model):", type="password")

# --- RAG Pipeline Setup ---

# Define PDF path and vector store details
pdf_file_name = "azure-ai-services-openai_may2024.pdf"
chroma_db_dir = "./azure-ai-services-openai_may2024_db"
collection_name = "azure-ai-services-openai_may2024"

# Initialize embedding model (ensure this model is available or downloaded locally)
# The model 'thenlper/gte-large' is typically downloaded on first use and cached.
embedding_model = SentenceTransformerEmbeddings(model_name='thenlper/gte-large')

@st.cache_resource
def load_vectorstore(_embedding_model, collection_name, persist_directory):
    if not os.path.exists(persist_directory):
        st.error(f"Vector store directory not found at {persist_directory}. "
                 "Please ensure the `azure-ai-services-openai_may2024_db` folder "
                 "and `azure-ai-services-openai_may2024.pdf` are in the same directory as app.py.")
        st.stop()
    
    # If the vector store was not persisted correctly in Colab, or for first run locally
    # you might need to re-create it. For now, assume it's persisted.
    # However, if the PDF is not present or DB is empty, this needs handling.

    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=_embedding_model
        )
        st.success("Vector store loaded successfully!")
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        st.warning("Attempting to re-create vector store from PDF. This might take a while.")
        # Fallback: if loading fails, try to re-create from PDF if available
        if not os.path.exists(pdf_file_name):
             st.error(f"PDF file not found at {pdf_file_name}. Cannot re-create vector store.")
             st.stop()
        
        pdf_loader = PyPDFLoader(pdf_file_name)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name='cl100k_base',
            chunk_size=512,
            chunk_overlap=16,
            disallowed_special=()
        )
        Azure_AI_PDF_chunks = pdf_loader.load_and_split(text_splitter)

        vectorstore = Chroma.from_documents(
            tqdm(Azure_AI_PDF_chunks, desc="Creating vector store"), # tqdm for Colab, but won't show in Streamlit
            embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        vectorstore.persist()
        st.success("Vector store re-created and loaded successfully.")
        return vectorstore

vectorstore = load_vectorstore(embedding_model, collection_name, chroma_db_dir)
retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 5})

# --- Q&A System Prompt ---
qna_system_message = '''
You are an Azure Solution Architect who advises customers on building Cloud AI services.
There is an upcoming meeting where you are enlisted for an interactive Q&A session where customers will ask you questions about the
Azure Open AI service and how this service could be used in their business.

User input will have the context required by you to answer user questions.
This context will begin with the token: ###Context.
The context contains references to specific portions of a document relevant to the user query.

Please answer user questions only using the context provided in the input.
Do not mention anything about the context in your final answer. Your response should only contain the answer to the question.

If the answer is not found in the context, respond "I don't know".
'''

qna_user_message_template = """
###Context
Here are some documents that are relevant to the question mentioned below.
{context}

###Question
{question}
"""

# --- Chat Interface ---
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
    try:
        client = Groq(api_key=groq_api_key)
        model_name = 'openai/gpt-oss-20b' # Using a common Groq model for demonstration
                                    # or specify openai/gpt-oss-20b if available on Groq

        with st.form("qna_form", clear_on_submit=False):
            user_query = st.text_area("Ask a question about Azure Open AI:", "")
            submitted = st.form_submit_button("Send")

        if submitted and user_query:
            st.write("Searching for relevant documents...")
            # Newer LangChain retrievers expose `.invoke` instead of `.get_relevant_documents`
            relevant_document_chunks = retriever.invoke(user_query)
            context_list = [d.page_content for d in relevant_document_chunks]
            context_for_query = " ".join(context_list)

            prompt = [
                {'role': 'system', 'content': qna_system_message},
                {'role': 'user', 'content': qna_user_message_template.format(
                     context=context_for_query,
                     question=user_query
                    )
                }
            ]

            st.write("Generating response...")
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model=model_name,
                    messages=prompt,
                    temperature=0
                )

                prediction = response.choices[0].message.content.strip()
            
            st.subheader("Azure Solution Architect:")
            st.write(prediction)

    except Exception as e:
        st.error(f"An error occurred with Groq API: {e}")
else:
    st.warning("Please enter your Groq API Key to start the chat.")
