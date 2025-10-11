import json
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- Configuration ---
INPUT_JSON_PATH = "./wikipedia_documents_with_embeddings.json"
OUTPUT_JSON_PATH = "./wikipedia_documents_regenerated.json"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v1"
# Using the higher precision as requested by the user's function
FLOATING_POINT_PRECISION = 22

# --- Global model for worker processes ---
# This will be initialized once per worker process to avoid repeated loading.
from sentence_transformers import SentenceTransformer
model = None

def init_worker():
    """Initializer for each worker process in the pool."""
    global model
        
    print(f"Process {os.getpid()}: Initializing model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

def regenerate_embedding_worker(doc):
    """
    Takes a document, generates a new embedding from its 'content' field,
    and returns the updated document with the embedding as a single string.
    """
    global model
    if "content" in doc and isinstance(doc["content"], str):
        # 1. Generate new embedding from the document's content
        new_embedding = model.encode(doc["content"])
        
        # 2. Format the new embedding to a high-precision, comma-separated string
        #    using the logic from the user-provided function.
        embedding_str_values = ",".join([f'{x:.{FLOATING_POINT_PRECISION}f}' for x in new_embedding])
        
        # 3. Wrap the string in brackets to match the C++ parser's expected format.
        final_embedding_string = f"[{embedding_str_values}]"
        
        # 4. Update the document's embedding field with the single string
        doc["embedding"] = final_embedding_string
        
    return doc

# --- Main Execution ---
if __name__ == "__main__":

    print(f"Loading data from '{INPUT_JSON_PATH}'...")
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        documents_to_process = json.load(f)["documents"]
    
    num_docs = len(documents_to_process)
    print(f"Loaded {num_docs} documents to process.")

    # 2. PARALLELISM & STATUS: Process documents in parallel.
    num_workers = 6
    print(f"Starting embedding re-generation using {num_workers} CPU cores...")
    
    processed_docs = []
    # The `initializer` function ensures each worker process loads the model once.
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        with tqdm(total=num_docs, desc="Generating new embeddings") as pbar:
            # Use imap_unordered for efficiency, processing documents as they complete.
            for result in pool.imap_unordered(regenerate_embedding_worker, documents_to_process):
                processed_docs.append(result)
                pbar.update(1)

    # 3. PERSISTENCE & OUTPUT: Write the new data to the output file.
    output_data = {"documents": processed_docs}

    print(f"\nWriting newly generated data to '{OUTPUT_JSON_PATH}'...")
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f)

