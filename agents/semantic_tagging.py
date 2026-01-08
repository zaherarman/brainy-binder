import json
import re

from llm.client import MistralClient
from llm.prompts import build_tagging_prompt
from db.session import get_session
from db.models import Document as dbDocument
from vectorstore.chroma_store import ChromaStore

class SemanticTaggingAgent():
    """
    Agent that generates semantic tags for documents using LLM.

    Tags are stored in the SQLite database and can be used for filtering.
    """
    def __init__(self, llm_client=None, chroma_store=None):
        """
        Initialize the semantic tagging agent.

        Args:
            llm_client: MistralClient instance
            chroma_store: ChromaStore instance (optional, for updating vector metadata)
        """
        self.llm_client = llm_client or MistralClient()
        self.chroma_store = chroma_store

    def run(self, document_path, document_id):
        """
        Generate and store semantic tags for a document.

        Args:
            document_path: Path to the document
            document_id: Database ID of the document

        Returns:
            Dictionary with tags and status
        """
        if not document_path and not document_id:
            raise ValueError("Must provide either document_path or document_id")

        with get_session() as session:
            if document_id:
                db_doc = session.query(dbDocument).filter(dbDocument.id == document_id).first()
            
            else:
                db_doc = session.query(dbDocument).filter(dbDocument.path == document_path).first()

            if not db_doc:
                raise ValueError(f"Document not found: {document_path or document_id}")

            title = db_doc.title
            doc_id = db_doc.id
            doc_path = db_doc.path

        if self.chroma_store:
            chunks = self.chroma_store.get_by_metadata(filter_dict={"document_id": doc_id}, limit=10)
        else:
            chunks = []

        if chunks:
            document_text = "\n\n".join(chunk.page_content for chunk in chunks[:3])
        
        else:
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    document_text = f.read(2000) 

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Could not read document content: {e}",
                    "tags": [],
                }

        messages = build_tagging_prompt(document_text, title)
        response = self.llm_client.chat(messages, temperature=0.5)

        tags = self._parse_tags(response)

        if not tags:
            return {
                "success": False,
                "error": "Could not extract tags from LLM response",
                "raw_response": response,
                "tags": [],
            }

        tags_str = ",".join(tags)

        with get_session() as session:
            db_doc = session.query(dbDocument).filter(dbDocument.id == doc_id).first()
            
            if db_doc:
                db_doc.tags = tags_str
                session.commit()

        return {
            "success": True,
            "document_id": doc_id,
            "document_path": doc_path,
            "title": title,
            "tags": tags,
            "tags_str": tags_str,
        }

    def parse_tags(self, response):
        """
        Parse tags from LLM response.

        Tries to extract a JSON array, falls back to a comma-separated parsing.

        Args:
            response: LLM response text

        Returns:
            List of tag strings
        """
        json_match = re.search(r"\[([^\]]+)\]", response)
        if json_match:
            try:
                tags = json.loads(json_match.group(0))
                
                if isinstance(tags, list):
                    return [str(tag).strip().lower() for tag in tags if tag]
            
            except json.JSONDecodeError:
                pass

        lines = response.strip().split("\n")
        
        for line in lines:
            if "," in line:
                line = re.sub(r"^(tags?:|topics?:)\s*", "", line, flags=re.IGNORECASE)
                tags = [tag.strip().strip("\"'").lower() for tag in line.split(",")]
                tags = [tag for tag in tags if tag and len(tag) > 1]
                
                if tags:
                    return tags

        return []

    def tag_all_documents(self, document_type):
        """
        Generate tags for all documents (or filtered by type).

        Args:
            document_type: Optional filter by document type

        Returns:
            Dictionary with statistics
        """
        stats = {"total": 0, "success": 0, "failed": 0, "errors": []}

        with get_session() as session:

            query = session.query(dbDocument)

            if document_type:
                query = query.filter(dbDocument.document_type == document_type)

            documents = query.all()
            stats["total"] = len(documents)

        for db_doc in documents:
            try:
                result = self.run(document_id=db_doc.id)
                
                if result["success"]:
                    stats["success"] += 1
                
                else:
                    stats["failed"] += 1
                    stats["errors"].append(
                        {"document_id": db_doc.id, "error": result.get("error", "Unknown error"),}
                    )

            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append({"document_id": db_doc.id, "error": str(e)})

        return stats

