import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

# Path to the persistent vector store
CHROMA_PATH = "chroma"

# Prompt template used to generate the LLM query with context
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    """
    Entry point for the CLI interface.
    Parses the input query from command-line arguments and sends it to the RAG pipeline.
    """
    parser = argparse.ArgumentParser(description="Run a RAG query with a local vector database and LLM.")
    parser.add_argument("query_text", type=str, help="The query text to retrieve relevant context and generate an answer.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    """
    Executes the Retrieval-Augmented Generation (RAG) pipeline.

    Args:
        query_text (str): The user's question.

    Steps:
        - Load the embedding function.
        - Load the Chroma vector database.
        - Search for top-k similar documents.
        - Format the prompt using the retrieved context.
        - Use the LLM to generate a response.
        - Print the response along with source document IDs.
    """
    # Load embedding model
    embedding_function = get_embedding_function()
    
    # Load persistent Chroma vector store
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Perform similarity search with top 5 results
    results = db.similarity_search_with_score(query_text, k=5)

    # Extract the content from retrieved documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Format the prompt with context and question
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Generate the answer using the LLM
    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    # Extract sources (document IDs) for transparency
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()
