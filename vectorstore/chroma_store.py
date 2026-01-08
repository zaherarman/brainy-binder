import chromadb

from chromadb import Settings as ChromaSettings
from langchain_core.documents import Document
from vectorstore.embeddings import EmbeddingService
from config import settings

class ChromaStore:
    """
    Wrapper for ChromaDB vector store with custom embeddings.

    Handles document storage, retrieval, and similarity search using sentence-transformers embeddings.
    """
    def __init__(self, persist_dir=None, collection_name=None, embedding_service=None):
        self.persist_dir = persist_dir or str(settings.chroma_db_dir)
        self.collection_name = collection_name or settings.chroma_collection_name
        self.embedding_service = embedding_service or EmbeddingService()
        self.client = chromadb.PersistentClient(path=self.persist_dir, settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True)) # On disk needed, not ra
        self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})

    def add_documents(self, documents, ids=None):
        """
        Add documents to vector store.

        Args:
            documents: A list of Document objects w page_content and metadata for each element.
            ids: An optional list of document ids (will generate if custom ids arent given)
        """
        if not documents:
            return
        
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        embeddings = self.embedding_service.embed_documents(texts)

        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]

        self.collection.add(embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids)

    def similarity_search(self, query, filter_dict=None, k=None):
        """
        Search for similar documents.

        Args:
            query: A text of your query
            k: Top k results to fetch from similarity search
            filter_dict: Optional filtering logic using metadata

        Returns:
            A list of Document objects with text content and and corresponding metadata
        """
        k = settings.top_k

        query_embedding = self.embedding_service.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_dict,
            include=['documents', 'metadatas', 'distances']
        )

        documents = []
    
        if results["documents"] and results["documents"][0]:
            for i, doc_text in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else {}

                if distance is not None:
                    metadata["similarity_score"] = 1 - distance

                documents.append(Document(page_content=doc_text, metadata=metadata))

        return documents
    
    def reset(self):
        """Helper function: Deletes all information in brainy_binder collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def count(self):
        """Helper function: Retreives the amount of documents in the collection"""
        return self.collection.count()
    
    def get_by_metadata(self, filter_dict, limit=100):
        """
        Get documents by metadata filtering.

        Args:
            filter_dict: Metadata filter
            limit: Max. number of results from search

        Returns:
            A list of Document objects
        """
        results = self.collection.get(where=filter_dict, limit=limit, include=["documents", "metadatas"])

        documents = []

        if results["documents"]:
            for i, doc_text in enumerate(results["documents"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                documents.append(Document(page_content=doc_text, metadata=metadata))

        return documents