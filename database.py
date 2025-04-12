import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma

# Default paths for data and vector database
DATA_PATH = 'books'
CHROMA_PATH = 'chroma'

def main():
    """
    Main entry point for ingesting documents into the Chroma vector database.

    - Optionally resets the database using --reset flag.
    - Loads documents from PDF files in a directory.
    - Splits documents into smaller chunks.
    - Adds chunks to the Chroma database if they don't already exist.
    """
    parser = argparse.ArgumentParser(description="Load and embed documents into a Chroma vector database.")
    parser.add_argument("--reset", action="store_true", help="Reset the vector database before ingesting.")
    args = parser.parse_args()

    if args.reset:
        print("Clearing Database")
        clear_database()

    documents = load_documents_from_directory()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents_from_directory(directory_path: str = DATA_PATH) -> list[Document]:
    """
    Loads all PDF documents from the specified directory.

    Args:
        directory_path (str): Directory containing PDF files.

    Returns:
        list[Document]: Loaded LangChain Document objects.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If the directory is empty.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory {directory_path} does not exist")

    if not os.listdir(directory_path):
        raise ValueError(f"Directory {directory_path} is empty")

    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()
    return documents

def split_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Splits documents into smaller chunks for better embedding and retrieval.

    Args:
        documents (list[Document]): List of input documents.
        chunk_size (int): Maximum number of characters in each chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        list[Document]: Chunked documents.

    Raises:
        ValueError: If the input document list is empty.
    """
    if not documents:
        raise ValueError("No documents to split")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    """
    Adds new chunks to the Chroma vector database if they are not already present.

    Args:
        chunks (list[Document]): List of document chunks to add.
    """
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    chunks_with_ids = calculate_chunk_ids(chunks)

    # Retrieve existing document IDs from the database
    existing_items = db.get(include=[])  # Only returns document IDs
    existing_ids = set(existing_items["ids"])

    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Filter out chunks that already exist in the DB
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("No new documents to add")

def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    """
    Generates unique IDs for each chunk based on its source, page number, and index.

    Example ID format: 'books/document.pdf:6:2'

    Args:
        chunks (list[Document]): List of chunks to assign IDs.

    Returns:
        list[Document]: Chunks with metadata["id"] field populated.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
        last_page_id = current_page_id

    return chunks

def clear_database():
    """
    Deletes the Chroma vector database directory.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
