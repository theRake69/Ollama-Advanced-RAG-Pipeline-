import os
import tempfile
from urllib import response

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

from click import prompt
from numpy import isin
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

import ollama
from sentence_transformers import CrossEncoder 

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

def get_vector_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url = "http://localhost:11434/api/embeddings",
        model_name = "nomic-embed-text:latest",
    )
    chroma_client = chromadb.PersistentClient(path="chroma_database")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"}
    )

def add_to_vector_collection(all_splits: list[Document], file_name: str):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        st.success("Data added to the vector store!")

def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results

def process_document(uploaded_file: UploadedFile) -> list[Document]:
    # store uploaded files as a temp file
    with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    loader = PyMuPDFLoader(temp_file_path)
    docs = loader.load()
    os.unlink(temp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )

    return text_splitter.split_documents(docs)

def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model = "llama3.2",
        stream = True,
        messages = [
            {
                "role": "system",
                "content": "system_prompt",
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}"
            }
        ]
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break

def re_rank_cross_encoders(query: str, documents: list[str]) -> tuple[str, list[int]]:
    # validate input types
    if not isinstance(query, str):
        raise ValueError("Query must be a string", f"Invalid Query: {query}")

    
    relevant_text = ""
    relevant_text_ids = []
    documents = [doc for doc in documents[0]]
    # st.write(documents)
    # st.error([doc[1] for doc in documents])

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    ranks = encoder_model.rank(query, documents, top_k=3)
    # st.write(ranks)

    for rank in ranks:
        # get the index of the document
        doc_index = rank["corpus_id"]
        # check if the doc_index is within bounds of documents
        if doc_index < 0 or doc_index >= len(documents):
            print(f"IndexError: doc_index {doc_index} is out of bounds for documents list.")
            continue
        relevant_text += documents[doc_index]
        relevant_text_ids.append(doc_index)

    return relevant_text, relevant_text_ids

def main():
    st.set_page_config(page_title="RAG Question Answer", layout="wide")
    st.header("RAG Question Answer")
    with st.sidebar:
        
        uploaded_file = st.file_uploader(
            "** Upload PDF files for QnA**", type=["pdf"],
            accept_multiple_files=False
        )

        process = st.button(
            "Process",
        )

        if uploaded_file and process:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, normalize_uploaded_file_name)

    user_prompt = st.text_area("**Ask a question related to your document:**")

    ask = st.button(
        "Ask"
    )

    if ask and user_prompt:
        results = query_collection(user_prompt)
        context_documents = results.get("documents", [0][0])

        # st.write(context_documents)

        if context_documents:
            relevant_text, relevant_text_ids = re_rank_cross_encoders(user_prompt, context_documents)
            response = call_llm(context=relevant_text, prompt=user_prompt)
            st.write_stream(response)
        else:
            st.error("No relevant documents found.")

        with st.expander("See retrieved Documents"):
            st.write(results)

        with st.expander("See most relevant document ids"):
            st.write(relevant_text_ids)
            st.write(relevant_text)

if __name__ == "__main__":
    main()