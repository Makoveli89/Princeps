from typing import Any, Dict, List

import networkx as nx
from keybert import KeyBERT


class ConceptGraphAgent:
    """
    An agent that builds and manages a concept graph from text content.
    """

    def __init__(self):
        """
        Initializes the ConceptGraphAgent.
        """
        self.graph: nx.DiGraph = nx.DiGraph()
        self.kw_model: KeyBERT = KeyBERT()

    def extract_concepts(self, content: str) -> List[Dict[str, Any]]:
        """
        Extracts key concepts from the given content using KeyBERT.

        Args:
            content: The text content to analyze.

        Returns:
            A list of concepts, where each concept is a dictionary
            containing the concept text and its relevance score.
        """
        keywords = self.kw_model.extract_keywords(
            content,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            use_mmr=True,
            diversity=0.7,
            top_n=15,
        )
        return [{"concept": kw[0], "relevance": kw[1]} for kw in keywords]

    def update_graph(self, doc_id: str, concepts: List[Dict[str, Any]]):
        """
        Updates the concept graph with concepts from a new document.

        Args:
            doc_id: The ID of the document being processed.
            concepts: A list of concepts extracted from the document.
        """
        if not concepts:
            return

        # Add document node
        self.graph.add_node(doc_id, type="document")

        # Add concept nodes and edges
        for item in concepts:
            concept = item["concept"]
            relevance = item["relevance"]

            # Add concept node if it doesn't exist
            if not self.graph.has_node(concept):
                self.graph.add_node(concept, type="concept")

            # Add edge from document to concept
            self.graph.add_edge(doc_id, concept, weight=relevance)

            # Add edges between co-occurring concepts
            for other_item in concepts:
                if item["concept"] != other_item["concept"]:
                    other_concept = other_item["concept"]
                    if not self.graph.has_edge(concept, other_concept):
                        self.graph.add_edge(concept, other_concept, weight=0)
                    self.graph[concept][other_concept]["weight"] += 1
