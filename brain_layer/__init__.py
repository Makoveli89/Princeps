"""
Princeps Brain Layer
====================

A modular AI knowledge management system recycled from Mothership/Lumina.

Sections:
- document_ingestion (01): PDF parsing, chunking
- activity_tracing (02): Agent run logging
- knowledge_distillation (03): NER, topics, summaries
- data_models (04): SQLAlchemy + pgvector schemas
- retrieval_systems (05): Multi-backend search
- promotion_contradiction (06): Priority scoring, conflicts
- agent_training (07): RL, A/B testing
- supabase_pgvector (08): Database integration

Created: December 26, 2024
"""

__version__ = "0.1.0"
__author__ = "Princeps"

# Submodule aliases for cleaner imports
# Note: Folder names start with numbers (01_, 02_, etc.) which aren't valid
# Python identifiers, so we use importlib to create aliases
import importlib

document_ingestion = importlib.import_module(".01_document_ingestion", "brain_layer")
activity_tracing = importlib.import_module(".02_activity_tracing", "brain_layer")
knowledge_distillation = importlib.import_module(".03_knowledge_distillation", "brain_layer")
data_models = importlib.import_module(".04_data_models", "brain_layer")
retrieval_systems = importlib.import_module(".05_retrieval_systems", "brain_layer")
promotion_contradiction = importlib.import_module(".06_promotion_contradiction", "brain_layer")
agent_training = importlib.import_module(".07_agent_training", "brain_layer")
supabase_pgvector = importlib.import_module(".08_supabase_pgvector", "brain_layer")
