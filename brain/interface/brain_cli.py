"""Brain CLI - Command-line interface."""

import argparse
import sys


class BrainCLI:
    """Command-line interface for Princeps Brain Layer."""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self):
        parser = argparse.ArgumentParser(description="Princeps Brain Layer CLI")
        subparsers = parser.add_subparsers(dest="command", help="Commands")

        # Init command
        init_parser = subparsers.add_parser("init", help="Initialize the database")
        init_parser.add_argument("--db-url", help="Database URL")

        # Ingest command
        ingest_parser = subparsers.add_parser("ingest", help="Ingest documents or repositories")
        ingest_parser.add_argument("path", help="Path to document or repository")
        ingest_parser.add_argument("--type", choices=["document", "repository"], default="document")
        ingest_parser.add_argument("--tenant", help="Tenant name")

        # Distill command
        distill_parser = subparsers.add_parser("distill", help="Distill documents")
        distill_parser.add_argument("--document-id", help="Document ID to distill")
        distill_parser.add_argument("--batch", action="store_true", help="Process all unanalyzed")

        # Metrics command
        metrics_parser = subparsers.add_parser("metrics", help="Show metrics")
        metrics_parser.add_argument("--json", action="store_true", help="Output as JSON")

        # Query command
        query_parser = subparsers.add_parser("query", help="Query the knowledge base")
        query_parser.add_argument("text", help="Query text")
        query_parser.add_argument("--limit", type=int, default=5, help="Number of results")

        return parser

    def run(self, args=None):
        parsed = self.parser.parse_args(args)

        if not parsed.command:
            self.parser.print_help()
            return 1

        handler = getattr(self, f"cmd_{parsed.command}", None)
        if handler:
            return handler(parsed)
        else:
            print(f"Unknown command: {parsed.command}")
            return 1

    def cmd_init(self, args):
        print("Initializing database...")
        from ..core.db import init_db

        init_db()
        print("Database initialized successfully!")
        return 0

    def cmd_ingest(self, args):
        print(f"Ingesting {args.type}: {args.path}")
        from ..ingestion import IngestService

        service = IngestService()
        if args.type == "document":
            result = service.ingest_document(args.path, tenant_name=args.tenant)
        else:
            result = service.ingest_repository(args.path, tenant_name=args.tenant)
        print(f"Result: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Documents: {result.documents_created}, Chunks: {result.chunks_created}")
        return 0 if result.success else 1

    def cmd_distill(self, args):
        print("Distilling documents...")
        from ..distillation import DistillationService

        service = DistillationService()
        if args.batch:
            results = service.distill_unanalyzed_documents()
            print(f"Processed {len(results)} documents")
        elif args.document_id:
            result = service.distill_document(args.document_id)
            print(f"Result: {'SUCCESS' if result.success else 'FAILED'}")
        return 0

    def cmd_metrics(self, args):
        print("Fetching metrics...")
        # Stub
        return 0

    def cmd_query(self, args):
        print(f"Querying: {args.text}")
        # Stub
        return 0


def cli_main():
    """Main entry point for CLI."""
    cli = BrainCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    cli_main()
