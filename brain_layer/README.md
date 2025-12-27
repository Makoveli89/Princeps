# Brain Layer - Component Taxonomy

This folder structure organizes reusable code components for the Princeps brain layer, mapped directly to the architectural analysis from the Document Ingestion, Parsing & Chunking PDF.

---

## Progress Overview

| Section | Folder | Status | Files | Size |
|---------|--------|--------|-------|------|
| §1 | `01_document_ingestion/` | ✅ Complete | 3 | 16KB |
| §2 | `02_activity_tracing/` | ✅ Complete | 2 | 28KB |
| §3 | `03_knowledge_distillation/` | ✅ Complete | 5 | ~85KB |
| §4 | `04_data_models/` | ✅ Complete | 2 | ~48KB |
| §5 | `05_retrieval_systems/` | ✅ Complete | 5 | 42KB |
| §6 | `06_promotion_contradiction/` | ✅ Complete | 4 | 54KB |
| §7 | `07_agent_training/` | ✅ Complete | 5 | 72KB |
| §8 | `08_supabase_pgvector/` | ✅ Complete | 5 | 51KB |

**Progress: 8/8 sections (100%) ✅**

**Total: ~31 files, ~400KB of recycled/created code**

---

## Folder Structure

| Folder | PDF Section | Purpose |
|--------|-------------|---------|
| `01_document_ingestion` | §1 - Document Ingestion (Parsing & Chunking) | PDF extraction, text chunking, metadata generation |
| `02_activity_tracing` | §2 - Activity Tracing & Logging (Runs/Decisions) | Agent run logging, decision tracking, JSONL fallback |
| `03_knowledge_distillation` | §3 - Knowledge Atom Distillation (Summaries & Links) | NER, topic modeling, summarization, concept graphs |
| `04_data_models` | §4 - Data Models (Postgres Schema Targets) | SQLAlchemy models, schema definitions, migrations |
| `05_retrieval_systems` | §5 - Retrieval Systems (Vector Search & Fallbacks) | Vector stores, embedding search, TF-IDF fallback |
| `06_promotion_contradiction` | §6 - Promotion & Contradiction Logic | Knowledge scoring, priority ranking, conflict detection |
| `07_agent_training` | §7 - Agent Training & Strategy Generation | RL agents, model training, A/B testing, versioning |
| `08_supabase_pgvector` | §8 - Supabase/pgvector Integration | Postgres adapters, pgvector utilities, Supabase clients |

---

## Source File Mapping

### 01_document_ingestion ✅
| File | Source Path | Description |
|------|-------------|-------------|
| `pdf_utils.py` | `Mothership/common/pdf_utils.py` | PDF extraction, chunking, metadata |
| `ingest_pdfs.py` | `Mothership/ingest_pdfs.py` | CLI script for bulk PDF ingestion |
| `librarian_tfidf.py` | `Lumina/src/services/librarian.py` | Lightweight TF-IDF text ingestion |

### 02_activity_tracing ✅
| File | Source Path | Description |
|------|-------------|-------------|
| `umi_client.py` | `Mothership/common/umi_client.py` | Unified Memory Interface client |
| `memory_system_improvements.py` | `Mothership/memory_system_improvements.py` | SQLite logging, learning storage |

### 03_knowledge_distillation ✅
| File | Source Path | Description |
|------|-------------|-------------|
| `librarian_agent.py` | `Mothership/agents/librarian.py` | Knowledge store with analysis pipeline |
| `summarization_agent.py` | `Mothership/agents/summarization_agent.py` | BART-based summarization |
| `concept_graph_agent.py` | `Mothership/agents/concept_graph_agent.py` | KeyBERT concepts, networkx graph |
| `ner_agent.py` | `Mothership/agents/ner_agent.py` | spaCy entity extraction |
| `topic_modeling_agent.py` | `Mothership/agents/topic_modeling_agent.py` | BERTopic classification |

### 04_data_models ✅
| File | Source Path | Description |
|------|-------------|-------------|
| `models.py` | **NEW** | 14 SQLAlchemy tables with pgvector |
| `knowledge_network.py` | `Mothership/knowledge_network.py` | KnowledgeNode/Edge dataclasses |

**Tables defined:**
- Documents, chunks, entities, topics, concepts, summaries
- Agent runs, knowledge nodes/edges/flows
- Concept graph, priority knowledge

### 05_retrieval_systems ✅
| File | Source Path | Description |
|------|-------------|-------------|
| `vector_store.py` | `Mothership/vector_store.py` | ChromaDB vector storage |
| `embedding_retriever.py` | `Mothership/embedding_prototype.py` | SBERT/TF-IDF retrieval |
| `tfidf_librarian.py` | `Lumina/src/services/librarian.py` | Pure Python TF-IDF |
| `unified_retriever.py` | **NEW** | Multi-backend retriever |
| `__init__.py` | **NEW** | Package exports |

**Backends (priority order):** pgvector → ChromaDB → TF-IDF → Heuristic

### 06_promotion_contradiction ✅
| File | Source Path | Description |
|------|-------------|-------------|
| `priority_scorer.py` | **NEW** | Composite scoring (confidence, recency, usage) |
| `contradiction_detector.py` | **NEW** | Semantic + rule-based conflict detection |
| `promotion_service.py` | **NEW** | Knowledge promotion and sharing |
| `__init__.py` | **NEW** | Package exports |

### 07_agent_training ✅
| File | Source Path | Description |
|------|-------------|-------------|
| `model_trainer.py` | `Mothership/learning/model_trainer.py` | Continuous retraining, versioning |
| `ab_testing.py` | `Mothership/learning/ab_testing.py` | Multi-variant testing framework |
| `feedback_loop.py` | `Mothership/learning/feedback_loop.py` | Performance monitoring, alerts |
| `reinforcement_learning.py` | `Mothership/agents/reinforcement_learning.py` | Q-learning agent |
| `__init__.py` | **NEW** | Package exports |

### 08_supabase_pgvector ✅
| File | Source Path | Description |
|------|-------------|-------------|
| `supabase_client.py` | **NEW** | Supabase REST API wrapper |
| `pg_utils.py` | **NEW** | psycopg2/SQLAlchemy pooling, pgvector |
| `__init__.py` | **NEW** | Package exports |
| `migrations/env.py` | **NEW** | Alembic environment config |
| `migrations/001_initial_schema.py` | **NEW** | Full schema with vector functions |

---

## Quick Start

```python
# Document Ingestion
from brain_layer.document_ingestion import extract_text_from_pdf, chunk_text

# Knowledge Distillation
from brain_layer.knowledge_distillation import SummarizationAgent, NERAgent

# Data Models
from brain_layer.data_models.models import Document, DocChunk, Base

# Retrieval
from brain_layer.retrieval_systems import create_retriever, UnifiedRetriever

# Promotion & Contradiction
from brain_layer.promotion_contradiction import PriorityScorer, ContradictionDetector

# Agent Training
from brain_layer.agent_training import ModelTrainer, ABTestFramework, ReinforcementLearning

# Database
from brain_layer.supabase_pgvector import SupabaseClient, create_postgres_pool
```

---

## Dependencies

**Core (always required):**
- Python 3.10+
- dataclasses, typing, logging (stdlib)

**Optional by section:**

| Section | Dependencies |
|---------|-------------|
| §1 Ingestion | `pypdf`, `tiktoken` |
| §3 Distillation | `transformers`, `spacy`, `bertopic`, `keybert`, `networkx` |
| §4 Models | `sqlalchemy`, `pgvector` |
| §5 Retrieval | `sentence-transformers`, `chromadb`, `psycopg2` |
| §7 Training | (none - pure Python) |
| §8 Database | `supabase`, `psycopg2-binary`, `sqlalchemy` |

Install all:
```bash
pip install pypdf tiktoken transformers spacy bertopic keybert networkx \
            sqlalchemy pgvector sentence-transformers chromadb \
            supabase psycopg2-binary
python -m spacy download en_core_web_sm
```

---

## Skill Reference

This project uses the `code-recycler` skill:
```
C:\Skills\code-recycler\SKILL.md
C:\Skills\code-recycler\PROJECT_BRIEF.md
C:\Skills\code-recycler\SESSION_RECAP.md
```

---

*Completed: December 26, 2024*
*Source repos: Mothership, Lumina, Claude-Mothership, Lumina_Clean*
