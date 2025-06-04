from keybert import KeyBERT

text = "John Doe from WHO visited Lagos on Jan 1, 2023. His invoice ID is INV-4421."

kw_model = KeyBERT()
keywords = kw_model.extract_keywords(text, top_n=5)

print("✅ Extracted Keywords:")
for kw, score in keywords:
    print(f"{kw} ({score:.2f})")
