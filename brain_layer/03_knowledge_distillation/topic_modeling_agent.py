from typing import List

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


class TopicModelingAgent:
    """
    An agent that performs topic modeling on text content.
    """

    def __init__(self):
        """
        Initializes the TopicModelingAgent.
        """
        vectorizer_model = CountVectorizer(stop_words="english")
        self.topic_model = BERTopic(vectorizer_model=vectorizer_model)

    def analyze_topics(self, content: str) -> list:
        """
        Analyzes the topics in the given content.

        Args:
            content: The text content to analyze.

        Returns:
            A list of topics.
        """
        # Short-circuit if content is empty or trivially small
        if not content or len(content.strip()) < 50:
            return []

        # Primary path: attempt BERTopic. Note that many dimensionality-reduction
        # backends fail on a single document. We catch and fallback gracefully.
        try:
            # Fit on a tiny batch with the single document; if the backend
            # cannot transform with a single sample, this will raise.
            topics, _ = self.topic_model.fit_transform([content])

            # Get the topic info
            topic_info = self.topic_model.get_topic_info()

            # Since we passed a single document, topics[0] is the assigned topic id
            document_topic = topic_info[topic_info["Topic"] == topics[0]]

            if not document_topic.empty:
                topic_data = document_topic.to_dict("records")[0]
                return [
                    {
                        "topic_id": int(topic_data["Topic"]),
                        "name": topic_data["Name"],
                        "keywords": topic_data["Representation"],
                    }
                ]
            return []
        except Exception:
            # Fallback: lightweight keyword-based "topic" when BERTopic cannot run
            try:
                # Prefer KeyBERT if available for higher quality keyphrases
                from keybert import KeyBERT  # lazy import

                kw_model = KeyBERT()
                keywords: List[str] = [
                    kw
                    for kw, _ in kw_model.extract_keywords(
                        content,
                        keyphrase_ngram_range=(1, 2),
                        stop_words="english",
                        top_n=10,
                    )
                ]

                return [
                    {
                        "topic_id": 0,
                        "name": "Keywords",
                        "keywords": keywords,
                    }
                ]
            except Exception:
                # Last-resort fallback: simple CountVectorizer frequencies
                vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10)
                try:
                    vec.fit_transform([content])
                    vocab = vec.get_feature_names_out().tolist()
                except Exception:
                    vocab = []

                return [
                    {
                        "topic_id": 0,
                        "name": "Keywords",
                        "keywords": vocab,
                    }
                ]
