import json
import ast
json_file_path = "./wikipedia_documents_regenerated.json"
input_file_path = "./input.txt" # record doc_id -> (titile, embedding str)
content_file_path = "./content.txt" # record doc_id -> file content

with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)["documents"]

print(len(data))

print(data[0])

        
embeddings = [doc["embedding"] for doc in data]
titles = [doc["title"] for doc in data]
content = [doc["content"] for doc in data]

doc_input = ""

for i, (title, embedding) in enumerate(zip(titles, embeddings)):
    embedding_str = "[" + ",".join([str(f'{x: .22f}') for x in ast.literal_eval(embedding)]) + "]"
    doc_input += f"{i}|{title}|{embedding_str}\n"
    
with open(input_file_path, "w") as f:
    f.write(doc_input)
    
doc_content = ""
for i, text in enumerate(content):
    cleaned_text = text.replace("\n", "\\n")
    doc_content += f"{i}|{cleaned_text}\n"

with open(content_file_path, "w", encoding="utf-8") as f:
    f.write(doc_content)