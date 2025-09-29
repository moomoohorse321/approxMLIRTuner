#!/usr/bin/env python3
"""
TinyLlama Interface for RAG Agent
Simplified: call Gemma models (1B / 4B) via HuggingFace, no dynamic selection
"""

from typing import List
from transformers import pipeline


gemma_1b = pipeline("text-generation", model="google/gemma-3-1b-it", device=0)
gemma_4b = pipeline("text-generation", model="google/gemma-3-4b-it", device=0)


class TinyLlamaInterface:
    """Interface for Gemma models (1B / 4B) via HuggingFace"""

    def __init__(self, default_model: str = "gemma3:1b", timeout: int = 60):
        self.default_model = default_model
        self.timeout = timeout
        print(f"🤖 Default model: {self.default_model}")
        print(f"🤖 Available models: ['gemma3:1b', 'gemma3:4b']")

    def generate_answer(self, model_name: str, question: str, context_documents: List[str]) -> str:
        """
        Generate answer using selected Gemma model with retrieved context

        Args:
            model_name: "gemma3:1b" or "gemma3:4b"
            question: The user's question
            context_documents: List of retrieved document texts

        Returns:
            Generated answer
        """
        # 合并上下文
        context = "\n\n".join(context_documents)

        # 构造 prompt
        prompt = f"""Based on the following context documents, please answer the question.

Context:
{context}

Question: {question}

Answer:"""

        return self._call_gemma_model(model_name, prompt)

    def _call_gemma_model(self, model_name: str, prompt: str) -> str:
        """实际调用 Gemma 模型"""
        if "1b" in model_name:
            resp = gemma_1b(prompt, max_length=256, do_sample=True, top_p=0.9)
        elif "4b" in model_name:
            resp = gemma_4b(prompt, max_length=256, do_sample=True, top_p=0.9)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return resp[0]["generated_text"].strip()


def main():
    """Quick test"""
    print("Testing TinyLlama Interface...")

    llm = TinyLlamaInterface()

    question = "When was Apollo 17 launched?"
    context = [
        "Apollo 17 was launched on December 7, 1972. It was the final mission of the Apollo program.",
        "The mission was commanded by Eugene Cernan and included Harrison Schmitt as the lunar module pilot.",
    ]

    print(f"\nQuestion: {question}")
    print(f"Context: {context}")

    answer = llm.generate_answer("gemma3:1b", question, context)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()