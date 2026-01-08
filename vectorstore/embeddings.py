from sentence_transformers import SentenceTransformer
import torch

from config import settings

class EmbeddingService:
    """
    Service for generating embeddings using sentence-transformers
    
    Provides methods to embed documents and queries and extract dim.
    """
    def __init__(self, model_name=None):
        self.model_name = model_name or settings.embedding_model_name
        self.model = None
        self.load_model()

    def load_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_name, device=device)
    
    def embed_documents(self, texts):
        """
        Embed a list of ducments
        
        Args:
            texts: a list of text strings to embed
        
        Returns:
            A list of embedding vectors 
        """
        if not texts:
            return False
        
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=32)

        return embeddings.tolist()
    
    def embed_query(self, text):
        """
        Embed a query
        
        Args:
            text: query text to embed
        
        Returns:
            An embedding vector of the query
        """
        embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        
        return embedding.tolist()
    
    def dimension(self):
        return self.model.get_sentence_embedding_dimension()