from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"

)

def embed_chunks(chunks):
    from langchain_community.vectorstores import FAISS
    from langchain.docstore.document import Document

    docs = [Document(page_content=chunk) for chunk in chunks]
    db = FAISS.from_documents(docs, embedding_model)
    return db
