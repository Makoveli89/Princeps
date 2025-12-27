# Claude Code Debug Brief: Princeps Brain Layer

## Location
`F:\Princeps\brain_layer\`

## What This Is
Recycled code from Mothership/Lumina repos organized into 8 sections. SQLAlchemy models + pgvector for a knowledge management brain layer.

## Priority Tasks

### 1. Verify Imports Work
```bash
cd F:\Princeps
python -c "from brain_layer.retrieval_systems import create_retriever"
python -c "from brain_layer.data_models.models import Document, Base"
```

### 2. Check for Missing Dependencies
Create `requirements.txt` if missing:
```
sqlalchemy>=2.0
psycopg2-binary
pgvector
sentence-transformers
chromadb
transformers
spacy
keybert
bertopic
networkx
tiktoken
pypdf
supabase
alembic
```

### 3. Lint All Python Files
```bash
pip install ruff
ruff check brain_layer/ --fix
```

Common issues to expect:
- Unused imports
- F-string without placeholders
- Missing type hints
- Line length violations

### 4. Type Check
```bash
pip install mypy
mypy brain_layer/ --ignore-missing-imports
```

### 5. Test Key Files
Priority files to verify run without syntax errors:
- `04_data_models/models.py` - SQLAlchemy models
- `05_retrieval_systems/unified_retriever.py` - Core retriever
- `08_supabase_pgvector/migrations/001_initial_schema.py` - Alembic migration

### 6. Known Issues
- Folder names have `01_` prefixes - valid Python but unusual
- Some relative imports assume package structure
- pgvector/chromadb are optional - code has fallbacks

## File Count
~31 files across 8 folders, ~400KB total

## After Debug
Push to GitHub for GPT/Gemini review.
