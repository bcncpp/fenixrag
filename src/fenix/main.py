import asyncio
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import json
from typing import Optional, List
import logging

from fenix.config.settings import settings
from database.connection import DatabaseManager
from services.document_service import DocumentService
from services.rag_service import RAGService
from data.sample_documents import SAMPLE_DOCUMENTS

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Rich console for beautiful output
console = Console()
app = typer.Typer(help="🚀 LangChain Gemini PGVector Document Search System")


@app.command()
def setup():
    """Initialize database and load sample documents"""
    
    console.print("[bold blue]Setting up LangChain Gemini PGVector System[/bold blue]")
    
    # Check API key
    if not settings.validate_api_key():
        console.print("[bold red]Invalid Google API key![/bold red]")
        console.print("Please set GOOGLE_API_KEY in your .env file")
        console.print("Get your key from: https://makersuite.google.com/app/apikey")
        raise typer.Exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Initialize database
        task1 = progress.add_task("Initializing database...", total=None)
        DatabaseManager.initialize_database()
        progress.update(task1, description="✅ Database initialized")
        
        # Load sample documents
        task2 = progress.add_task("Loading sample documents...", total=None)
        asyncio.run(load_sample_documents())
        progress.update(task2, description="✅ Sample documents loaded")
    
    console.print("\n[bold green]Setup completed successfully![/bold green]")
    console.print("\nNext steps:")
    console.print("• Run: [bold cyan]fenix search[/bold cyan] for interactive search")
    console.print("• Run: [bold cyan]fenix rag[/bold cyan] for Q&A mode")


@app.command()
def search():
    """Interactive document search"""
    
    console.print("🔍 [bold blue]Interactive Document Search[/bold blue]")
    console.print("Commands: search <query> | filter <category> <query> | quit")
    
    asyncio.run(interactive_search())


@app.command()
def rag():
    """Interactive RAG Q&A"""
    
    console.print("🤖 [bold blue]RAG Question & Answer Mode[/bold blue]")
    console.print("Ask questions and get AI-powered answers based on your documents")
    
    asyncio.run(interactive_rag())


@app.command()
def add(content: str, title: Optional[str] = None, source: Optional[str] = None):
    """Add a single document"""
    
    async def _add_doc():
        doc_service = DocumentService()
        doc_id = await doc_service.add_document(
            content=content,
            title=title,
            source=source or "manual_input"
        )
        console.print(f"✅ Added document: {doc_id}")
    
    asyncio.run(_add_doc())


@app.command()
def batch_add(file_path: str):
    """Add documents from JSON file"""
    
    try:
        with open(file_path, 'r') as f:
            documents_data = json.load(f)
        
        async def _batch_add():
            doc_service = DocumentService()
            doc_ids = await doc_service.add_documents_batch(documents_data)
            console.print(f"Added {len(doc_ids)} documents from {file_path}")
        
        asyncio.run(_batch_add())
        
    except Exception as e:
        console.print(f" Error: {e}")
        raise typer.Exit(1)


async def load_sample_documents():
    """Load sample documents into the system"""
    
    doc_service = DocumentService()
    
    # Check if documents already exist
    existing_docs = await doc_service.similarity_search("Python programming", limit=1)
    if existing_docs:
        console.print("Sample documents already loaded, skipping...")
        return
    
    # Load sample documents
    await doc_service.add_documents_batch(SAMPLE_DOCUMENTS)
    console.print(f"Loaded {len(SAMPLE_DOCUMENTS)} sample documents")


async def interactive_search():
    """Interactive search loop"""
    
    doc_service = DocumentService()
    
    while True:
        try:
            user_input = console.input("\n🔎 Enter command: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            parts = user_input.split(' ', 2)
            command = parts[0].lower()
            
            if command == "search" and len(parts) > 1:
                query = ' '.join(parts[1:])
                
                with console.status(f"Searching for: {query}"):
                    docs_with_scores = await doc_service.similarity_search(query, limit=5)
                
                if docs_with_scores:
                    table = Table(title=f"Search Results for: {query}")
                    table.add_column("Score", justify="right", style="cyan")
                    table.add_column("Title", style="bold")
                    table.add_column("Content", style="dim")
                    table.add_column("Category", justify="center")
                    
                    for doc, score in docs_with_scores:
                        category = doc.metadata.get('category', 'unknown') if doc.metadata else 'unknown'
                        content_preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
                        
                        table.add_row(
                            f"{score:.3f}",
                            doc.title or "Untitled",
                            content_preview,
                            category
                        )
                    
                    console.print(table)
                else:
                    console.print("No documents found")
            
            elif command == "filter" and len(parts) > 2:
                category = parts[1]
                query = ' '.join(parts[2:])
                
                with console.status(f"Filtering {category} for: {query}"):
                    docs_with_scores = await doc_service.similarity_search(
                        query, 
                        limit=5, 
                        filters={"category": category}
                    )
                
                if docs_with_scores:
                    console.print(f"📂 Results in '{category}' category:")
                    for i, (doc, score) in enumerate(docs_with_scores, 1):
                        console.print(f"  {i}. [Score: {score:.3f}] {doc.content}")
                else:
                    console.print(f"No documents found in '{category}' category")
            
            else:
                console.print("Invalid command. Use: search <query> | filter <category> <query> | quit")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"Error: {e}")
    
    console.print("👋 Search session ended")


async def interactive_rag():
    """Interactive RAG Q&A loop"""
    
    try:
        rag_service = RAGService(use_llm=True)
    except ValueError as e:
        console.print(f"❌ {e}")
        console.print("Running in context-only mode...")
        rag_service = RAGService(use_llm=False)
    
    console.print("\n💡 Ask questions about your documents (type 'quit' to exit)")
    
    while True:
        try:
            question = console.input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            with console.status(f"Processing: {question}"):
                result = await rag_service.query(question)
            
            # Display answer
            console.print(Panel(
                result["answer"],
                title="🤖 Answer",
                border_style="green"
            ))
            
            # Display sources
            if result["sources"]:
                console.print(f"\n📚 Sources (Confidence: {result['confidence']:.3f}):")
                for i, source in enumerate(result["sources"], 1):
                    console.print(f"  {i}. [{source['confidence']:.3f}] {source['title'] or 'Untitled'}")
                    console.print(f"     {source['content_preview']}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f" Error: {e}")
    
    console.print("👋 RAG session ended")


if __name__ == "__main__":
    app()
