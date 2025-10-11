from typing import Dict
content_path = "./content.txt"

documents: Dict[int, str] = {}
with open(content_path, 'r', encoding='utf-8') as f:
    for line in f:
        # Split the line only at the first pipe character.
        # This prevents errors if the document content itself contains a '|'.
        parts = line.strip().split('|', 1)
        
        if len(parts) == 2:
            doc_id_str, content_str = parts
            doc_id = int(doc_id_str)
            # Restore the newline characters that were escaped.
            original_content = content_str.replace('\\n', '\n')
            documents[doc_id] = original_content
        else:
            raise ValueError("Parsed content format incorrect")

print(documents[0])