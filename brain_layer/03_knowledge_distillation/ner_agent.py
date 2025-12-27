"""
NER Agent - Named Entity Recognition
Extracts entities from text using spaCy.
"""

import spacy


class NERAgent:
    def __init__(self, model="en_core_web_lg"):
        """
        Initializes the NER Agent with a spaCy model.
        """
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Spacy model '{model}' not found. Please run 'python -m spacy download {model}'")
            raise

    def extract_entities(self, text: str) -> dict:
        """
        Extracts named entities from the given text.
        """
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        return entities
