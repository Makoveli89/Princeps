"""
PDF Ingestion Script for Brain Layer
Ingests PDF documents from D:\ drive into the Librarian knowledge base
"""

import sys
from datetime import datetime
from pathlib import Path

import PyPDF2

# Add Mothership to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.librarian import LibrarianAgent


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text content from PDF file"""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        print(f"❌ Error reading {pdf_path.name}: {e}")
        return ""


def ingest_pdfs(pdf_paths: list[str]):
    """Ingest multiple PDFs into Librarian"""
    print("\n" + "=" * 70)
    print("📚 PDF INGESTION TO BRAIN LAYER")
    print("=" * 70 + "\n")

    # Initialize Librarian
    print("🔧 Initializing Librarian Agent...")
    librarian = LibrarianAgent()

    stats_before = librarian.get_statistics()
    print(f"📊 Starting entries: {stats_before['total_entries']}\n")

    ingested_count = 0
    failed_count = 0

    for pdf_path_str in pdf_paths:
        pdf_path = Path(pdf_path_str)

        if not pdf_path.exists():
            print(f"⚠️  Skipping: {pdf_path.name} (not found)")
            failed_count += 1
            continue

        print(f"\n📄 Processing: {pdf_path.name}")
        print(f"   Size: {pdf_path.stat().st_size / 1024:.2f} KB")

        # Extract text
        print("   📖 Extracting text...")
        text = extract_text_from_pdf(pdf_path)

        if not text:
            print("   ❌ No text extracted, skipping")
            failed_count += 1
            continue

        word_count = len(text.split())
        print(f"   ✅ Extracted {word_count:,} words")

        # Add to Librarian
        try:
            entry_id = librarian.store_agent_knowledge(
                agent_name="PDFImporter",
                task=f"Import PDF: {pdf_path.name}",
                content=text[:50000],  # Limit to first 50K chars
                tags=["pd", "document", "imported"],
                metadata={
                    "file_path": str(pdf_path),
                    "file_name": pdf_path.name,
                    "file_size": pdf_path.stat().st_size,
                    "word_count": word_count,
                    "ingested_at": datetime.now().isoformat(),
                },
            )
            print(f"   ✅ Ingested as entry: {entry_id}")
            ingested_count += 1

        except Exception as e:
            print(f"   ❌ Failed to ingest: {e}")
            failed_count += 1

    # Final stats
    stats_after = librarian.get_statistics()
    print("\n" + "=" * 70)
    print("📊 INGESTION COMPLETE")
    print("=" * 70)
    print(f"✅ Successfully ingested: {ingested_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📚 Total entries now: {stats_after['total_entries']}")
    print(f"📈 New entries added: {stats_after['total_entries'] - stats_before['total_entries']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # PDFs from D:\ drive
    pdf_files = [
        r"D:\(Un)making AI Magic_ a Design Taxonomy.pd",
        r"D:\``DecisionTime''_ A Configurable Framework for Reproducible Human - AI Decision - Making Experiments.pd",
        r"D:\``Information - Backward but Sex - Forward''_ Navigating Masculinity towards Intimate Well - being.pd",
        r"D:\2602945.2602955.pd",
        r"D:\26339137221078005.pdf",
    ]

    ingest_pdfs(pdf_files)
