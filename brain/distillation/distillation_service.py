"""
Princeps Brain Layer - Knowledge Distillation & Analysis
=========================================================

Distills ingested content into structured knowledge atoms:
- Summaries (extractive/abstractive)
- Named entities (spaCy/regex)
- Topics/keywords (KeyBERT/RAKE)
- Key concepts with relevance scores

Usage:
    from brain.distillation import DistillationService

    service = DistillationService()
    result = service.distill_document(document_id)
"""

import logging
import re
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..core.db import (
    get_default_tenant_id,
    get_engine,
    get_or_create_operation,
    get_session,
    mark_operation_failed,
    mark_operation_started,
    mark_operation_success,
)
from ..core.models import (
    Document,
    DocumentConcept,
    DocumentEntity,
    DocumentSummary,
    DocumentTopic,
    OperationStatusEnum,
    OperationTypeEnum,
)

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for distillation pipeline."""

    max_summary_length: int = 500
    executive_summary_sentences: int = 3
    detailed_summary_sentences: int = 10
    entity_confidence_threshold: float = 0.5
    max_entities_per_doc: int = 100
    deduplicate_entities: bool = True
    max_topics: int = 5
    max_keywords_per_topic: int = 10
    keyword_ngram_range: tuple[int, int] = (1, 3)
    max_concepts: int = 20
    concept_relevance_threshold: float = 0.1
    batch_size: int = 10
    continue_on_error: bool = True
    max_errors_per_batch: int = 5
    use_transformers: bool = True
    fallback_to_heuristics: bool = True


@dataclass
class DistillationResult:
    """Result of a distillation operation."""

    success: bool
    operation_id: str | None = None
    document_id: str | None = None
    summaries_created: int = 0
    entities_created: int = 0
    topics_created: int = 0
    concepts_created: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: int = 0
    model_used: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class SummarizationService:
    """Generate document summaries."""

    def __init__(self, config: DistillationConfig):
        self.config = config
        self._model = None
        self._method = None

    @property
    def available_method(self) -> str:
        if self._method is None:
            if self.config.use_transformers:
                try:
                    from transformers import pipeline

                    self._model = pipeline(
                        "summarization", model="facebook/bart-large-cnn", device=-1
                    )
                    self._method = "transformer"
                except:
                    self._method = "heuristic"
            else:
                self._method = "heuristic"
        return self._method

    def summarize(
        self, text: str, one_sentence: bool = True, executive: bool = True, detailed: bool = False
    ) -> dict[str, str | None]:
        result = {
            "one_sentence": None,
            "executive": None,
            "detailed": None,
            "model_used": self.available_method,
        }
        if not text or len(text.strip()) < 50:
            return result

        text = re.sub(r"\s+", " ", text).strip()

        if self.available_method == "transformer" and self._model:
            try:
                if one_sentence:
                    result["one_sentence"] = self._model(
                        text[:3000], max_length=60, min_length=20, do_sample=False
                    )[0]["summary_text"]
                if executive:
                    result["executive"] = self._model(
                        text[:3000], max_length=150, min_length=50, do_sample=False
                    )[0]["summary_text"]
                if detailed:
                    result["detailed"] = self._model(
                        text[:3000], max_length=300, min_length=100, do_sample=False
                    )[0]["summary_text"]
                return result
            except:
                pass

        # Heuristic fallback
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        if sentences:
            if one_sentence:
                result["one_sentence"] = sentences[0][: self.config.max_summary_length]
            if executive:
                result["executive"] = " ".join(sentences[: self.config.executive_summary_sentences])
            if detailed:
                result["detailed"] = " ".join(sentences[: self.config.detailed_summary_sentences])
        result["model_used"] = "heuristic"
        return result


class EntityExtractionService:
    """Extract named entities."""

    def __init__(self, config: DistillationConfig):
        self.config = config
        self._nlp = None
        self._available = None

    @property
    def is_available(self) -> bool:
        if self._available is None:
            try:
                import spacy

                self._nlp = spacy.load("en_core_web_sm")
                self._available = True
            except:
                self._available = False
        return self._available

    def extract(self, text: str) -> list[dict[str, Any]]:
        if not text:
            return []

        if self.is_available:
            entities = []
            for ent in self._nlp(text[:100000]).ents:
                if len(ent.text.strip()) >= 2:
                    entities.append(
                        {
                            "text": ent.text.strip(),
                            "label": ent.label_,
                            "start_char": ent.start_char,
                            "end_char": ent.end_char,
                            "confidence": 0.85,
                            "model_used": "spacy",
                        }
                    )
            return self._deduplicate(entities)[: self.config.max_entities_per_doc]

        # Regex fallback
        entities = []
        patterns = {
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "URL": r'https?://[^\s<>"{}|\\^`\[\]]+',
            "TECH": r"\b(?:Python|JavaScript|TypeScript|React|PostgreSQL|Docker|AWS|API|ML|AI)\b",
        }
        for label, pattern in patterns.items():
            for m in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(
                    {
                        "text": m.group(),
                        "label": label,
                        "start_char": m.start(),
                        "end_char": m.end(),
                        "confidence": 0.7,
                        "model_used": "regex",
                    }
                )
        return self._deduplicate(entities)[: self.config.max_entities_per_doc]

    def _deduplicate(self, entities: list[dict]) -> list[dict]:
        seen = {}
        for ent in entities:
            key = (ent["text"].lower(), ent["label"])
            if key in seen:
                seen[key]["frequency"] = seen[key].get("frequency", 1) + 1
            else:
                ent["frequency"] = 1
                seen[key] = ent
        return sorted(seen.values(), key=lambda x: x.get("frequency", 1), reverse=True)


class TopicExtractionService:
    """Extract topics and keywords."""

    def __init__(self, config: DistillationConfig):
        self.config = config
        self._keybert = None
        self._method = None

    @property
    def available_method(self) -> str:
        if self._method is None:
            try:
                from keybert import KeyBERT

                self._keybert = KeyBERT()
                self._method = "keybert"
            except:
                self._method = "heuristic"
        return self._method

    def extract(self, text: str) -> list[dict[str, Any]]:
        if not text or len(text.strip()) < 100:
            return []

        if self.available_method == "keybert" and self._keybert:
            try:
                keywords = self._keybert.extract_keywords(
                    text,
                    keyphrase_ngram_range=self.config.keyword_ngram_range,
                    stop_words="english",
                    top_n=self.config.max_keywords_per_topic,
                )
                if keywords:
                    return [
                        {
                            "topic_id": 0,
                            "name": keywords[0][0],
                            "keywords": [kw[0] for kw in keywords],
                            "probability": sum(kw[1] for kw in keywords) / len(keywords),
                            "model_used": "keybert",
                        }
                    ]
            except:
                pass

        # Heuristic
        import collections

        words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        counter = collections.Counter(words)
        keywords = [w for w, _ in counter.most_common(self.config.max_keywords_per_topic)]
        if keywords:
            return [
                {
                    "topic_id": 0,
                    "name": keywords[0],
                    "keywords": keywords,
                    "probability": 0.5,
                    "model_used": "heuristic",
                }
            ]
        return []


class ConceptExtractionService:
    """Extract key concepts with relevance scores."""

    def __init__(self, config: DistillationConfig):
        self.config = config
        self._keybert = None
        self._method = None

    @property
    def available_method(self) -> str:
        if self._method is None:
            try:
                from keybert import KeyBERT

                self._keybert = KeyBERT()
                self._method = "keybert"
            except:
                self._method = "heuristic"
        return self._method

    def extract(self, text: str) -> list[dict[str, Any]]:
        if not text or len(text.strip()) < 50:
            return []

        if self.available_method == "keybert" and self._keybert:
            try:
                keywords = self._keybert.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 3),
                    stop_words="english",
                    top_n=self.config.max_concepts,
                    use_mmr=True,
                    diversity=0.7,
                )
                return [
                    {"concept": phrase, "relevance": float(score), "model_used": "keybert"}
                    for phrase, score in keywords
                    if score >= self.config.concept_relevance_threshold
                ]
            except:
                pass

        # Heuristic
        import collections

        words = text.lower().split()
        bigrams = [" ".join(words[i : i + 2]) for i in range(len(words) - 1)]
        counter = collections.Counter(bigrams)
        concepts = []
        for phrase, count in counter.most_common(self.config.max_concepts * 2):
            if any(w in phrase for w in ["the", "and", "for", "with"]):
                continue
            relevance = min(1.0, count / 10)
            if relevance >= self.config.concept_relevance_threshold:
                concepts.append(
                    {"concept": phrase, "relevance": relevance, "model_used": "heuristic"}
                )
        return concepts[: self.config.max_concepts]


class DistillationService:
    """Main distillation service for knowledge extraction."""

    def __init__(self, config: DistillationConfig = None, db_url: str = None):
        self.config = config or DistillationConfig()
        self._engine = None

        self.summarizer = SummarizationService(self.config)
        self.entity_extractor = EntityExtractionService(self.config)
        self.topic_extractor = TopicExtractionService(self.config)
        self.concept_extractor = ConceptExtractionService(self.config)

        if db_url:
            from sqlalchemy import create_engine

            self._engine = create_engine(db_url)

    @property
    def engine(self):
        if self._engine is None:
            self._engine = get_engine()
        return self._engine

    def distill_document(
        self,
        document_id: str,
        tenant_name: str = None,
        generate_summary: bool = True,
        extract_entities: bool = True,
        extract_topics: bool = True,
        extract_concepts: bool = True,
    ) -> DistillationResult:
        """Distill a single document into knowledge atoms."""
        result = DistillationResult(success=False, document_id=document_id)
        start_time = datetime.utcnow()

        try:
            with get_session() as session:
                document = session.query(Document).filter(Document.id == document_id).first()
                if not document:
                    result.errors.append(f"Document not found: {document_id}")
                    return result

                tenant_id = str(document.tenant_id)
                op_inputs = {
                    "document_id": document_id,
                    "summary": generate_summary,
                    "entities": extract_entities,
                    "topics": extract_topics,
                    "concepts": extract_concepts,
                }
                operation, created = get_or_create_operation(
                    session,
                    tenant_id,
                    OperationTypeEnum.ANALYSIS,
                    op_inputs,
                    document_id=document_id,
                )
                result.operation_id = str(operation.id)

                if not created and operation.status == OperationStatusEnum.SUCCESS:
                    result.success = True
                    return result

                mark_operation_started(session, operation.id)

                try:
                    text = document.content

                    if generate_summary:
                        existing = (
                            session.query(DocumentSummary)
                            .filter(DocumentSummary.document_id == document.id)
                            .first()
                        )
                        if not existing:
                            summaries = self.summarizer.summarize(
                                text, True, True, len(text) > 5000
                            )
                            session.add(
                                DocumentSummary(
                                    tenant_id=tenant_id,
                                    document_id=document.id,
                                    one_sentence=summaries.get("one_sentence"),
                                    executive=summaries.get("executive"),
                                    detailed=summaries.get("detailed"),
                                    model_used=summaries.get("model_used"),
                                )
                            )
                            result.summaries_created = 1
                            result.model_used["summarization"] = summaries.get("model_used")

                    if extract_entities:
                        if (
                            session.query(DocumentEntity)
                            .filter(DocumentEntity.document_id == document.id)
                            .count()
                            == 0
                        ):
                            entities = self.entity_extractor.extract(text)
                            for ent in entities:
                                session.add(
                                    DocumentEntity(
                                        tenant_id=tenant_id,
                                        document_id=document.id,
                                        text=ent["text"][:500],
                                        label=ent["label"],
                                        start_char=ent.get("start_char"),
                                        end_char=ent.get("end_char"),
                                        confidence=ent.get("confidence"),
                                        frequency=ent.get("frequency", 1),
                                    )
                                )
                            result.entities_created = len(entities)
                            if entities:
                                result.model_used["entity_extraction"] = entities[0].get(
                                    "model_used"
                                )

                    if extract_topics:
                        if (
                            session.query(DocumentTopic)
                            .filter(DocumentTopic.document_id == document.id)
                            .count()
                            == 0
                        ):
                            topics = self.topic_extractor.extract(text)
                            for t in topics:
                                session.add(
                                    DocumentTopic(
                                        tenant_id=tenant_id,
                                        document_id=document.id,
                                        topic_id=t["topic_id"],
                                        name=t.get("name"),
                                        keywords=t.get("keywords", []),
                                        probability=t.get("probability"),
                                    )
                                )
                            result.topics_created = len(topics)
                            if topics:
                                result.model_used["topic_extraction"] = topics[0].get("model_used")

                    if extract_concepts:
                        if (
                            session.query(DocumentConcept)
                            .filter(DocumentConcept.document_id == document.id)
                            .count()
                            == 0
                        ):
                            concepts = self.concept_extractor.extract(text)
                            for c in concepts:
                                session.add(
                                    DocumentConcept(
                                        tenant_id=tenant_id,
                                        document_id=document.id,
                                        concept=c["concept"][:255],
                                        relevance=c["relevance"],
                                    )
                                )
                            result.concepts_created = len(concepts)
                            if concepts:
                                result.model_used["concept_extraction"] = concepts[0].get(
                                    "model_used"
                                )

                    document.is_analyzed = True
                    mark_operation_success(
                        session,
                        operation.id,
                        {
                            "summaries": result.summaries_created,
                            "entities": result.entities_created,
                            "topics": result.topics_created,
                            "concepts": result.concepts_created,
                        },
                    )
                    result.success = True

                except Exception as e:
                    # Log detailed error internally but avoid exposing specifics via result.errors
                    logger.exception("Error during document distillation for document_id=%s", document_id)
                    result.errors.append("Internal error during document distillation")
                    mark_operation_failed(session, operation.id, str(e), traceback.format_exc())
                    session.rollback()

        except Exception as e:
            # Log outer failure details; return a generic error message to callers
            logger.exception("Unexpected failure in distill_document for document_id=%s", document_id)
            result.errors.append("Distillation failed due to an internal error")

        finally:
            result.duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        return result

    def distill_unanalyzed_documents(
        self, tenant_name: str = None, limit: int = None
    ) -> list[DistillationResult]:
        """Process all un-analyzed documents."""
        results = []
        error_count = 0

        with get_session() as session:
            query = session.query(Document).filter(Document.is_analyzed == False)
            if tenant_name:
                tenant_id = get_default_tenant_id(session)
                query = query.filter(Document.tenant_id == tenant_id)
            query = query.limit(limit or self.config.batch_size)

            for doc in query.all():
                if error_count >= self.config.max_errors_per_batch:
                    break
                try:
                    result = self.distill_document(str(doc.id))
                    results.append(result)
                    if not result.success:
                        error_count += 1
                except Exception as e:
                    error_count += 1
                    results.append(
                        DistillationResult(success=False, document_id=str(doc.id), errors=[str(e)])
                    )

        return results


def main():
    """CLI entry point."""
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Distillation Service")
    parser.add_argument("--document-id", help="Document UUID")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    service = DistillationService()

    if args.batch:
        results = service.distill_unanalyzed_documents(limit=args.limit)
        print(
            f"Processed {len(results)} documents, {sum(1 for r in results if r.success)} successful"
        )
    elif args.document_id:
        result = service.distill_document(args.document_id)
        print(
            f"{'SUCCESS' if result.success else 'FAILED'}: summaries={result.summaries_created}, entities={result.entities_created}"
        )


if __name__ == "__main__":
    main()
