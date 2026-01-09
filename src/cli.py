from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import settings
from .db.session import init_db
from .ingestion.pipeline import IngestionPipeline
from .rag.answer_engine import AnswerEngine
from .agents.semantic_tagging import SemanticTaggingAgent
from .vectorstore.chroma_store import ChromaStore

app = typer.Typer(name="brainy-binder", help="Privacy-first local AI knowledge assistant", add_completion=False)
console = Console()

@app.command()
def ingest(
    data_dir: Path = typer.Option(None, "--data-dir"),
    reset_index: bool = typer.Option(False, "--reset-index"),
):    
    
    """Ingest documents from a directory into the knowledge base."""

    
    console.print(Panel.fit("[bold cyan]Brainy Binder - Document Ingestion[/bold cyan]", border_style="cyan"))

    init_db()
    data_directory = Path(data_dir) if data_dir else settings.data_dir
    
    if not data_directory.exists():
        console.print(f"[red]Error: Data directory does not exist: {data_directory}[/red]")
        console.print("[yellow]Tip: Create the directory and add some documents first.[/yellow]")
        raise typer.Exit(code=1)

    try:
        pipeline = IngestionPipeline(data_dir=data_directory, reset_index=reset_index)
        stats = pipeline.run()

        if stats["files_processed"] > 0:
            console.print("\n[bold green]✓ Ingestion successful![/bold green]")

        else:
            console.print("\n[yellow]No new documents were ingested.[/yellow]")

    except Exception as e:
        console.print(f"\n[red]Error during ingestion: {e}[/red]")
        raise typer.Exit(code=1)

@app.command()
def query(
    question=typer.Argument(..., help="Question to ask"),
    top_k=typer.Option(None, "--top-k", "-k", help="Number of source documents to retrieve"),
    show_sources: bool = typer.Option(True, "--show-sources/--no-sources", help="Show source documents"),
):
    
    """Ask a question and get an answer from your knowledge base."""

    init_db()
    console.print(f"\n[cyan]Question:[/cyan] {question}\n")

    try:
        engine = AnswerEngine(top_k=top_k)

        with console.status("[bold cyan]Searching and generating answer...[/bold cyan]"):
            answer, sources = engine.answer_question(question, top_k=top_k)

        console.print(Panel(answer, title="[bold green]Answer[/bold green]", border_style="green"))

        if show_sources and sources:
            console.print("\n[bold cyan]Sources:[/bold cyan]\n")

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("#", style="dim", width=3)
            table.add_column("Source", style="cyan")
            table.add_column("Score", justify="right", style="green")
            table.add_column("Preview", style="dim")

            for i, doc in enumerate(sources, 1):
                source = doc.metadata.get("source_path", "Unknown")
                source = Path(source).name if source != "Unknown" else source
                score = doc.metadata.get("similarity_score", 0.0)
                preview = doc.page_content[:80].replace("\n", " ") + "..."
                table.add_row(str(i), source, f"{score:.3f}", preview)

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)

@app.command()
def summarize(
    path=typer.Option(None, "--path", "-p", help="Path to the document to summarize"),
    doc_id=typer.Option(None, "--doc-id", "-i", help="Database ID of the document"),
):
    
    """Summarize a specific document."""

    if not path and not doc_id:
        console.print("[red]Error: Must provide either --path or --doc-id[/red]")
        raise typer.Exit(code=1)

    init_db()

    try:
        engine = AnswerEngine()

        with console.status("[bold cyan]Generating summary...[/bold cyan]"):
            summary = engine.summarize_document(document_path=path, document_id=doc_id)

        doc_info = engine.get_document_info(document_path=path, document_id=doc_id)
        title = doc_info["title"] if doc_info else "Document"

        console.print(Panel(summary, title=f"[bold green]Summary: {title}[/bold green]", border_style="green"))

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)

@app.command()
def tag_doc(
    path=typer.Option(None, "--path", "-p", help="Path to the document to tag"),
    doc_id=typer.Option(None, "--doc-id", "-i", help="Database ID of the document"),
):
    
    """Generate semantic tags for a document."""

    if not path and not doc_id:
        console.print("[red]Error: Must provide either --path or --doc-id[/red]")
        raise typer.Exit(code=1)

    init_db()

    try:
        agent = SemanticTaggingAgent()

        with console.status("[bold cyan]Generating tags...[/bold cyan]"):
            result = agent.run(document_path=path, document_id=doc_id)

        if result["success"]:
            console.print(f"\n[bold green]✓ Tags generated for: {result['title']}[/bold green]\n")
            console.print(f"[cyan]Tags:[/cyan] {', '.join(result['tags'])}")

        else:
            console.print(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
            raise typer.Exit(code=1)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)

@app.command()
def chat():

    """Start an interactive chat session."""

    init_db()

    console.print(
        Panel.fit(
            "[bold cyan]Brainy Binder - Interactive Chat[/bold cyan]\n"
            "Ask questions about your knowledge base.\n"
            "Type 'exit', 'quit', or 'q' to end the session.",
            border_style="cyan",
        )
    )

    engine = AnswerEngine()

    while True:
        console.print()
        question = console.input("[bold yellow]You:[/bold yellow] ").strip()

        if not question:
            continue

        if question.lower() in ["exit", "quit", "q"]:
            console.print("\n[cyan]Goodbye! [/cyan]")
            break

        try:
            with console.status("[bold cyan]Thinking...[/bold cyan]"):
                answer, sources = engine.answer_question(question)

            console.print(f"\n[bold green]Brainy Binder:[/bold green] {answer}\n")

            if sources:
                console.print("[dim]Sources:[/dim]")

                for i, doc in enumerate(sources[:3], 1):
                    source = doc.metadata.get("source_path", "Unknown")
                    source = Path(source).name if source != "Unknown" else source
                    score = doc.metadata.get("similarity_score", 0.0)
                    console.print(f"  [dim][{i}] {source} (score: {score:.3f})[/dim]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

@app.command()
def list_docs(
    doc_type=typer.Option(None, "--type", "-t", help="Filter by document type (note, pdf, bookmark)"),
    limit=typer.Option(50, "--limit", "-n", help="Maximum number of documents to show"),
):
    
    """List indexed documents."""

    init_db()

    try:
        engine = AnswerEngine()
        documents = engine.list_documents(document_type=doc_type, limit=limit)

        if not documents:
            console.print("[yellow]No documents found.[/yellow]")
            return

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", style="dim", width=6)
        table.add_column("Type", style="cyan", width=10)
        table.add_column("Title", style="green")
        table.add_column("Tags", style="yellow")

        for doc in documents:
            table.add_row(
                str(doc["id"]),
                doc["document_type"],
                doc["title"][:60],
                doc["tags"][:40] if doc["tags"] else "",
            )

        console.print(f"\n[bold]Found {len(documents)} documents:[/bold]\n")
        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)

@app.command()
def info():
    
    """Show configuration and basic stats."""

    console.print(Panel.fit("[bold cyan]Brainy Binder - System Information[/bold cyan]", border_style="cyan"))

    info_table = Table(show_header=False, box=None)
    info_table.add_column("Setting", style="cyan")
    info_table.add_column("Value", style="green")

    info_table.add_row("LLM Endpoint", settings.llm_base_url)
    info_table.add_row("LLM Model", settings.llm_model_name)
    info_table.add_row("Embedding Model", settings.embedding_model_name)
    info_table.add_row("Data Directory", str(settings.data_dir))
    info_table.add_row("ChromaDB Directory", str(settings.chroma_db_dir))
    info_table.add_row("SQLite Database", str(settings.sqlite_db_path))
    info_table.add_row("Default Top-K", str(settings.top_k))
    info_table.add_row("Chunk Size", str(settings.chunk_size))
    info_table.add_row("Chunk Overlap", str(settings.chunk_overlap))

    console.print("\n[bold]Configuration:[/bold]\n")
    console.print(info_table)

    try:
        init_db()

        chroma_store = ChromaStore()
        vector_count = chroma_store.count()

        engine = AnswerEngine()
        documents = engine.list_documents(limit=10000)
        doc_count = len(documents)

        console.print(f"\n[bold]Statistics:[/bold]")
        console.print(f"  Indexed documents: [green]{doc_count}[/green]")
        console.print(f"  Vector chunks: [green]{vector_count}[/green]")

    except Exception as e:
        console.print(f"\n[yellow]Could not load statistics: {e}[/yellow]")


def main():
    app()


if __name__ == "__main__":
    main()