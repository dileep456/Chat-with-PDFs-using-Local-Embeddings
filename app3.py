import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer


# Initialize embedding and LLM models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Correct wrapper
qa_model_name = "distilbert-base-uncased-distilled-squad"
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)


def get_pdf_text(pdf_docs):
    """Extract text from PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


def get_text_chunks(text):
    """Split the extracted text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)


def create_vector_store(text_chunks):
    """Create a vector store using FAISS and local embeddings."""
    docs = [Document(page_content=chunk) for chunk in text_chunks]
    
    # Use FAISS with the proper embedding model
    vector_store = FAISS.from_documents(documents=docs, embedding=embedding_model)
    
    # Save the vector store locally for reuse
    vector_store.save_local("faiss_index")
    return vector_store


def load_vector_store():
    """Load the FAISS vector store from the local file."""
    return FAISS.load_local(
        "faiss_index", 
        embeddings=embedding_model, 
        allow_dangerous_deserialization=True
    )




def get_answer_from_llm(context, question):
    """Use the local language model to generate an answer."""
    answer = qa_pipeline({'context': context, 'question': question})
    return answer['answer']


def process_user_query(user_question):
    """Process the user's input question and return an answer."""
    vector_store = load_vector_store()

    # Perform a similarity search to retrieve the most relevant context
    relevant_docs = vector_store.similarity_search(user_question)
    context = " ".join([doc.page_content for doc in relevant_docs])

    if context.strip():
        answer = get_answer_from_llm(context, user_question)
    else:
        answer = "No relevant context found in the documents."
    
    return answer


def main():
    """Main function to set up the Streamlit app."""
    st.set_page_config(page_title="Chat with PDF using Local Embeddings 💁")
    st.header("Chat with PDF using Local Embeddings 💁")

    # Sidebar for uploading and processing PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process"):
            with st.spinner("Processing PDFs..."):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    create_vector_store(text_chunks)
                    st.success("PDFs processed and vector store created!")
                else:
                    st.error("Please upload at least one PDF.")

    # Main section for user query
    user_question = st.text_input("Ask a Question:")
    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Fetching answer..."):
                answer = process_user_query(user_question)
                st.write(f"**Answer:** {answer}")
        else:
            st.error("Please enter a question.")


if __name__ == "__main__":
    main()
