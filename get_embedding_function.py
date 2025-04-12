from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function():
    """
    Initializes and returns an embedding function using the OllamaEmbeddings class.

    This function loads a text embedding model (in this case, 'nomic-embed-text')
    which will later be used to convert text into vector representations for tasks
    like semantic search or retrieval.

    Returns:
        OllamaEmbeddings: An object capable of generating vector embeddings from text.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings