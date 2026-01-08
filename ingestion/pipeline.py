from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from config import settings
from db.session import get_session
from db.models import Document as dbDocument
from vectorstore.chroma_store import ChromaStore
from ingestion.loaders import discover_documents, load_document
from ingestion.chunking import chunk_documents

console = Console()

class IngestionPipeline:
    def __init__(self, reset_index, data_dir=None):
        self.data_dir = data_dir or settings.data_dir
        self.chroma_store = ChromaStore()
        self.reset_index = reset_index # Ensures a clean ingestion state

    def run(self):
        stats = {
            "files_discovered": 0,
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "documents_index": 0
        }

        if self.reset_index:
            console.print("[yellow]Resetting index...[/yellow]")

            self.chroma_store.reset()
            self.clear_database()

            console.print("[green]Index reset complete![/green]")

        console.print(f"[cyan]Discovering documents in {self.data_dir}...[/cyan]")
        filepaths = discover_documents(self.data_dir)
        stats["files_discovered"] = len(filepaths)

        if not filepaths:
            console.print(f"[red]No documents found in {self.data_dir}![/red]")
            return stats
        
        console.print(f"[green]Found {len(filepaths)} files in {self.data_dir}[/green]")

        with Progress(
            SpinnerColumn(), 
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Processing documents...", total=len(filepaths))

            for filepath in filepaths:
                try:
                    if self.is_filed_indexed(filepath):
                        progress.update(task, advance=1)
                        continue
                
                    documents = load_document(filepath)

                    if not documents:
                        stats["files_failed"] += 1
                        progress.update(task, advance=1)
                        continue

                    chunks = chunk_documents(documents)
                    stats["chunks_created"] += len(chunks)

                    doc_id = self.store_document_metadata(filepath, documents[0])

                    for chunk in chunks:
                        chunk.metadata["document_id"] = doc_id

                    self.chroma_store.add_documents(chunks)

                    stats["files_processed"] += 1
                    stats["documents_index"] += 1

                except Exception as e:
                    console.print(f"[red]Error processing {filepath} due to error {e}[/red]")
                    stats["files_failed"] += 1
                
                progress.update(task, advance=1)

        console.print("\n[bold green]Ingestion complete![/bold green]")
        console.print(f"   > Files discovered: {stats['files_discovered']}")
        console.print(f"   > Files processed: {stats['files_processed']}")
        console.print(f"   > Files failed: {stats['files_failed']}")
        console.print(f"   > Chunks created: {stats['chunks_created']}")
        console.print(f"   > Total vectors in store: {self.chroma_store.count()}")     
        
        return stats

    def is_filed_indexed(self, filepath):
        with get_session() as session:
            exists = session.query(dbDocument).filter(dbDocument.path == str(filepath)).first() is not None
            return exists

    def store_document_metadata(self, filepath, doc):
        with get_session() as session:
            db_doc = dbDocument(
                path = str(filepath),
                document_type = doc.metadata.get("document_type",'unknown'),
                title=doc.metadata.get("title", filepath.stem),
                description=doc.metadata.get("description", ""),
                tags=""
            )

            session.add(db_doc)
            session.flush()
            doc_id = db_doc.id

        return doc_id

    def clear_database(self):
        with get_session() as session:
            session.query(dbDocument).delete()