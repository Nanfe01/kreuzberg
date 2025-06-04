from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    from gliner import GLiNER
except ImportError:
    GLiNER = None

try:
    from keybert import KeyBERT
except ImportError:
    KeyBERT = None


@dataclass
class Entity:
    type: str
    text: str
    start: int
    end: int


def extract_entities_and_keywords(
    text: str,
    extract_entities: bool = False,
    extract_keywords: bool = False,
    keyword_count: int = 10,
    custom_entity_patterns: Optional[dict[str, str]] = None
) -> Tuple[Optional[List[Entity]], Optional[List[Tuple[str, float]]]]:
    entities: List[Entity] = []
    keywords: List[Tuple[str, float]] = []

    # ✅ Built-in NER using GLiNER
    if extract_entities:
        if GLiNER is None:
            raise ImportError("GLiNER is not installed. Install it with: pip install gliner")
        model = GLiNER.from_pretrained("urchade/gliner_small")
        predicted = model.predict(text, return_offsets=True)
        allowed_types = {"PERSON", "ORGANIZATION", "LOCATION", "DATE", "EMAIL", "PHONE"}
        entities.extend([
            Entity(type=e['label'], text=e['text'], start=e['start'], end=e['end'])
            for e in predicted if e['label'] in allowed_types
        ])

    # ✅ Custom regex-based entities
    if custom_entity_patterns:
        import re
        for label, pattern in custom_entity_patterns.items():
            for match in re.finditer(pattern, text):
                entities.append(
                    Entity(type=label, text=match.group(), start=match.start(), end=match.end())
                )

    # ✅ Keyword extraction using KeyBERT
    if extract_keywords:
        if KeyBERT is None:
            raise ImportError("KeyBERT is not installed. Install it with: pip install keybert")
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(text, top_n=keyword_count)

    return entities if entities else None, keywords if keywords else None
