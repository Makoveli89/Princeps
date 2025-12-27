import logging
import os
from typing import Dict

from transformers import pipeline

logger = logging.getLogger(__name__)

DEFAULT_SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
FALLBACK_SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"


class SummarizationAgent:
    """
    An agent that generates summaries of text content.
    """

    def __init__(self, model_name: str | None = None):
        """
        Initializes the SummarizationAgent.
        """
        preferred_model = model_name or os.getenv("LIBRARIAN_SUMMARIZER_MODEL", DEFAULT_SUMMARIZATION_MODEL)
        fallback_model = os.getenv("LIBRARIAN_SUMMARIZER_FALLBACK", FALLBACK_SUMMARIZATION_MODEL)
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")

        self.summarizer = self._load_pipeline(preferred_model, fallback_model, hf_token)

    def _load_pipeline(self, model_name: str, fallback_model: str, hf_token: str | None):
        """Attempt to load the requested model, falling back if necessary."""
        tried_models = []
        candidates = [model_name]
        if fallback_model and fallback_model != model_name:
            candidates.append(fallback_model)

        for candidate in candidates:
            tried_models.append(candidate)
            try:
                return pipeline("summarization", model=candidate, token=hf_token)
            except Exception as exc:
                logger.warning(
                    "Summarization model %s unavailable (%s).",
                    candidate,
                    exc,
                )

        logger.warning(
            "Unable to load any summarization model (%s). Falling back to simple truncation.",
            ", ".join(tried_models),
        )
        return None

    def generate_summaries(self, content: str) -> Dict[str, str]:
        """
        Generates multi-level summaries for the given content.

        Args:
            content: The text content to summarize.

        Returns:
            A dictionary containing summaries of different lengths.
        """
        # Ensure content is not too short for summarization
        if len(content.split()) < 50:
            return {
                "one_sentence": content[:150],
                "executive": content,
            }

        if not self.summarizer:
            return {
                "one_sentence": content[:150],
                "executive": content[:1000],
            }

        try:
            # Executive summary (medium length)
            executive_summary = self.summarizer(content, max_length=150, min_length=50, do_sample=False)

            # One-sentence summary (short)
            one_sentence_summary = self.summarizer(content, max_length=60, min_length=20, do_sample=False)

            return {
                "one_sentence": one_sentence_summary[0]["summary_text"],
                "executive": executive_summary[0]["summary_text"],
            }
        except Exception as e:
            logger.warning("Error during summarization; falling back to truncation: %s", e)
            return {
                "one_sentence": content[:150],
                "executive": content[:1000],
            }
