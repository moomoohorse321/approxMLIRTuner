#!/usr/bin/env python3
import json
import tempfile
import subprocess
import os
import time
from typing import List, Tuple
from llm import LLMManager
from approxMLIR import ApproxMLIRSDK
from paths import TUNER, ROOT, BENCH
from sentence_transformers import SentenceTransformer
SDK: ApproxMLIRSDK = ApproxMLIRSDK("./binary", "./MLIR", ROOT)

def check_answer_correct(generated_answer: str, expected_answers: List[str]) -> bool:
    generated_lower = generated_answer.lower()

    for expected in expected_answers:
        if expected.lower() in generated_lower:
            return True

    return False

class DirectembeddingRetriever:
    def __init__(self, embedding_binary_path: str = None):
        self.embedding_binary_path = embedding_binary_path or os.path.abspath("embedding_file")
        self.documents = []
        self.document_ids = []
        self.document_titles = []
        self.document_embeddings = []
        self.is_loaded = False
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v1")

    def load_documents_from_json(self, json_file_path: str):
        print(f"Loading documents from {json_file_path}...")
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.documents = [doc["content"] for doc in data["documents"]]
        self.document_titles = [doc["title"] for doc in data["documents"]]
        self.document_ids = [f"doc_{i}" for i in range(len(data["documents"]))]
        self.document_embeddings = [doc["embedding"] for doc in data["documents"]]
        self.is_loaded = True

        print(f"✅ Loaded {len(self.documents)} documents")
        print(f"📊 Total characters: {sum(len(doc) for doc in self.documents):,}")
        print(f"📊 Embedding dimension: {len(self.document_embeddings[0])}")
        print(
            f"📊 Average length: {sum(len(doc) for doc in self.documents) // len(self.documents):,} chars/doc"
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        if not self.is_loaded:
            raise ValueError(
                "Documents not loaded. Call load_documents_from_json() first."
            )

        query_embedding = self.model.encode([query])[0].tolist()
        query_embedding_str = ",".join([str(x) for x in query_embedding])

        doc_input = ""
        for i, (title, embedding) in enumerate(
            zip(self.document_titles, self.document_embeddings)
        ):
            embedding_str = "[" + ",".join([str(x) for x in embedding]) + "]"
            doc_input += f"{i}|{title}|{embedding_str}\n"

        cmd = [self.embedding_binary_path, query_embedding_str, str(top_k)]

        result = subprocess.run(
            cmd,
            input=doc_input,
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
        )

        if result.returncode != 0:
            raise RuntimeError(f"embedding C program failed: {result.stderr}")

        retrieved_docs = self._parse_embedding_output(result.stdout, top_k)

        return retrieved_docs
    
    def _parse_embedding_output(self, output: str, top_k: int) -> List[Tuple[str, float, str, str]]:
        lines = output.strip().split("\n")
        retrieved_docs = []

        for line in lines:
            if line.startswith("Rank ") and "Doc " in line:
                doc_start = line.find("Doc ") + 4
                doc_end = line.find(" (Score:")
                doc_index = int(line[doc_start:doc_end])

                score_start = line.find("Score: ") + 7
                score_end = line.find(") -")
                score = float(line[score_start:score_end])

                title_start = line.find(') - "') + 5
                title_end = line.rfind('"')
                title = line[title_start:title_end]

                if doc_index < len(self.documents):
                    content = self.documents[doc_index]
                    doc_id = f"doc_{doc_index}"
                    retrieved_docs.append((doc_id, score, content, title))

        return retrieved_docs[:top_k]



class DirectRAGAgent:

    def __init__(
        self,
        documents_json_path: str,
        embedding_binary_path: str = None,
    ):
        self.documents_json_path = documents_json_path
        self.embedding_binary_path = embedding_binary_path or os.path.abspath("/binary/embedding.exec")

        self.retriever = DirectembeddingRetriever(self.embedding_binary_path)

        self.model_map = {
            1: "270m",  # Gemma3 270M (292MB, smaller model)
            2: "1b",  # Gemma3 1B (815MB, larger model)
        }
        self.llm = LLMManager(self.model_map.values())

        self.retriever.load_documents_from_json(documents_json_path)

        print("✅ Direct RAG Agent initialized successfully!")
        print(f"🤖 Available models: {list(self.model_map.values())}")

    def select_model_based_on_scores(self, avg_score: float) -> str:

        input_score = int(avg_score * 100)

        model_id = SDK.get_knob_val(1, input_score)
        selected_model = self.model_map.get(model_id, self.model_map[2])

        return self.model_map[2]  

    def answer_question(self, question: str, top_k: int = 10) -> dict:
       
        embedding_start_time = time.time()
        retrieved_docs = self.retriever.retrieve(question, top_k)
        embedding_time = time.time() - embedding_start_time

        if not retrieved_docs:
            return {
                "question": question,
                "answer": "No relevant documents found.",
                "retrieved_docs": [],
                "retrieval_count": 0,
            }

        print(f"✅ Retrieved {len(retrieved_docs)} documents:")
        for i, (doc_id, score, content, title) in enumerate(retrieved_docs):
            print(f"  {i+1}. {title} (Score: {score:.4f})")

        avg_score = sum(score for _, score, _, _ in retrieved_docs) / len(
            retrieved_docs
        )
        selected_model = self.select_model_based_on_scores(avg_score)

        print(f"🤖 Generating answer with {selected_model}...")
        start_time = time.time()
        prompt = f"""
        You are a QA agent you must answer the question based on these documents:{retrieved_docs}.
        Question: {question}. Answer: 
                    """
        answer = self.llm.generate(prompt, selected_model)
        llm_generation_time = time.time() - start_time

        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": [
                (doc_id, score, title) for doc_id, score, _, title in retrieved_docs
            ],
            "retrieval_count": len(retrieved_docs),
            "embedding_retrieval_time": embedding_time,
            "llm_generation_time": llm_generation_time,
            "avg_embedding_score": avg_score,
            "selected_model": selected_model,
        }


def main():
    agent = DirectRAGAgent(
        documents_json_path="wikipedia_documents_with_embeddings_first_100.json", embedding_binary_path="/home/yimu3/approxMLIRTuner/examples/LLM_embedding/binary/embedding.exec"
    )

    with open("enhanced_custom_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    test_questions = dataset["questions"]
    expected_answers = dataset["answers"]

    correct_count = 0
    all_results = []

    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Question {i}/{len(test_questions)}: {question}")
        print(f"🎯 Expected Answer: {expected_answers[i-1]}")
        print("-" * 50)

        result = agent.answer_question(question, top_k=5)
        generated_answer = result["answer"]

        is_correct = check_answer_correct(generated_answer, expected_answers[i - 1])
        if is_correct:
            correct_count += 1

        print(f"🤖 Generated Answer: {generated_answer}")
        print(f"⏱️  embedding Retrieval Time: {result['embedding_retrieval_time']:.3f} seconds")
        print(f"⏱️  LLM Generation Time: {result['llm_generation_time']:.2f} seconds")
        print(f"📊 Retrieved {result['retrieval_count']} documents")

        if result["retrieved_docs"]:
            print("📚 Retrieved documents:")
            for j, (doc_id, score, title) in enumerate(result["retrieved_docs"]):
                print(f"  {j+1}. {title} (Score: {score:.4f})")

        question_result = {
            "question_id": i,
            "question": question,
            "expected_answers": expected_answers[i - 1],
            "generated_answer": generated_answer,
            "is_correct": is_correct,
            "embedding_retrieval_time": result["embedding_retrieval_time"],
            "llm_generation_time": result["llm_generation_time"],
            "retrieved_documents": [
                {"rank": j + 1, "doc_id": doc_id, "title": title, "score": score}
                for j, (doc_id, score, title) in enumerate(result["retrieved_docs"])
            ],
            "retrieval_count": result["retrieval_count"],
        }
        all_results.append(question_result)

    accuracy = correct_count / len(test_questions)
    total_embedding_time = sum(result["embedding_retrieval_time"] for result in all_results)
    total_llm_time = sum(result["llm_generation_time"] for result in all_results)
    avg_embedding_time = total_embedding_time / len(test_questions) if test_questions else 0
    avg_llm_time = total_llm_time / len(test_questions) if test_questions else 0

    evaluation_results = {
        "metadata": {
            "total_questions": len(test_questions),
            "correct_answers": correct_count,
            "accuracy": accuracy,
            "total_embedding_time": total_embedding_time,
            "average_embedding_time": avg_embedding_time,
            "total_llm_time": total_llm_time,
            "average_llm_time": avg_llm_time,
            "retrieval_method": "embedding",
            "top_k": 5,
        },
        "questions": all_results,
    }

    output_file = "rag_evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

    total_time = total_embedding_time + total_llm_time
    print(f"\nTUNER_OUTPUT: time={total_time:.6f} accuracy={accuracy:.6f}")
    return total_time, accuracy

if __name__ == "__main__":
    main()
