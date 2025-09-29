#!/usr/bin/env python3
"""
直接使用100个Wikipedia文档的RAG Agent
不需要动态获取Wikipedia内容，直接使用预加载的文档库
"""

import json
import tempfile
import subprocess
import os
import time
from typing import List, Tuple
from tinyllama_interface import TinyLlamaInterface
from approxMLIR import ApproxMLIRSDK
from paths import TUNER, ROOT, BENCH
SDK: ApproxMLIRSDK = ApproxMLIRSDK("./binary", "./MLIR", ROOT)

def check_answer_correct(generated_answer: str, expected_answers: List[str]) -> bool:
    """
    检查生成的答案是否包含期望的答案

    Args:
        generated_answer: 生成的答案
        expected_answers: 期望的答案列表

    Returns:
        是否包含期望答案
    """
    generated_lower = generated_answer.lower()

    for expected in expected_answers:
        if expected.lower() in generated_lower:
            return True

    return False


class DirectBM25Retriever:
    """直接使用预加载文档的BM25检索器"""

    def __init__(self, bm25_binary_path: str = None):
        self.bm25_binary_path = bm25_binary_path or os.path.abspath("bm25_file")
        self.documents = []
        self.document_ids = []
        self.document_titles = []
        self.is_loaded = False

    def load_documents_from_json(self, json_file_path: str):
        """从JSON文件加载文档"""
        print(f"Loading documents from {json_file_path}...")

        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = data["documents"]
        self.documents = [doc["content"] for doc in docs]
        self.document_titles = [doc["title"] for doc in docs]
        self.document_ids = [f"doc_{i}" for i in range(len(docs))]
        self.is_loaded = True

        print(f"✅ Loaded {len(self.documents)} documents")
        print(f"📊 Total characters: {sum(len(doc) for doc in self.documents):,}")
        print(
            f"📊 Average length: {sum(len(doc) for doc in self.documents) // len(self.documents):,} chars/doc"
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        检索最相关的文档

        Args:
            query: 查询问题
            top_k: 返回的文档数量

        Returns:
            List of tuples (document_id, score, document_text)
        """
        if not self.is_loaded:
            raise ValueError(
                "Documents not loaded. Call load_documents_from_json() first."
            )

        # 创建临时文件存储文档
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt"
        ) as temp_file:
            for doc in self.documents:
                temp_file.write(doc + "\n")
            temp_file_path = temp_file.name

        # 打印输入信息
        print(f"🔍 Query: {query}")
        print(f"📁 Temp file: {temp_file_path}")
        print(f"📊 Temp file size: {os.path.getsize(temp_file_path)} bytes")
        print(f"📄 Temp file lines: {sum(1 for line in open(temp_file_path))} lines")

        # 调用BM25 C程序
        cmd = [self.bm25_binary_path, temp_file_path, query]
        print(f"🚀 Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
            errors="ignore",
        )

        if result.returncode != 0:
            raise RuntimeError(f"BM25 C program failed: {result.stderr}")

        # 解析输出
        retrieved_docs = self._parse_bm25_output(result.stdout, top_k)

        # 删除临时文件
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            print(f"🗑️  Temp file deleted: {temp_file_path}")

        return retrieved_docs

    def _parse_bm25_output(
        self, output: str, top_k: int
    ) -> List[Tuple[str, float, str]]:
        """解析BM25输出"""
        lines = output.strip().split("\n")
        retrieved_docs = []

        for line in lines:
            if line.startswith("Rank ") and "Doc " in line:
                # 解析格式: "Rank 1: Doc 0 (Score: 2.5664) - "content""
                # 提取文档索引
                doc_start = line.find("Doc ") + 4
                doc_end = line.find(" (Score:")
                doc_index = int(line[doc_start:doc_end])

                # 提取分数
                score_start = line.find("Score: ") + 7
                score_end = line.find(") -")
                score = float(line[score_start:score_end])

                # 提取内容
                content_start = line.find(') - "') + 5
                content_end = line.rfind('"')
                content = line[content_start:content_end]

                # 获取文档标题
                if doc_index < len(self.document_titles):
                    title = self.document_titles[doc_index]
                    doc_id = f"doc_{doc_index}"
                    retrieved_docs.append((doc_id, score, content, title))

        return retrieved_docs[:top_k]


class DirectRAGAgent:

    def __init__(
        self,
        documents_json_path: str,
        bm25_binary_path: str = None,
    ):
        self.documents_json_path = documents_json_path
        self.bm25_binary_path = bm25_binary_path or os.path.abspath("/binary/bm25.exec")

        # 初始化组件
        self.retriever = DirectBM25Retriever(self.bm25_binary_path)

        # 初始化LLM接口（不使用动态选择）
        self.llm = TinyLlamaInterface()

        # 定义模型映射
        self.model_map = {
            1: "gemma3:270m",  # Gemma3 270M (292MB, smaller model)
            2: "gemma3:1b",  # Gemma3 1B (815MB, larger model)
        }

        # 加载文档
        self.retriever.load_documents_from_json(documents_json_path)

        print("✅ Direct RAG Agent initialized successfully!")
        print(f"🤖 Available models: {list(self.model_map.values())}")

    def select_model_based_on_scores(self, avg_score: float) -> str:
        """
        根据BM25平均分数选择模型

        Args:
            avg_score: BM25检索文档的平均分数

        Returns:
            选择的模型名称
        """

        # 将平均分数转换为整数（乘以1000保留3位小数精度）
        input_score = int(avg_score * 1000)

        model_id = SDK.get_knob_val(1, input_score)
        selected_model = self.model_map.get(model_id, self.model_map[2])

        return self.model_map[2]  

    def answer_question(self, question: str, top_k: int = 10) -> dict:
        """
        回答问题

        Args:
            question: 用户问题
            top_k: 检索的文档数量

        Returns:
            包含答案和检索信息的字典
        """
        print(f"\n🔍 Processing question: {question}")

        # 1. 检索相关文档
        print("📚 Retrieving relevant documents...")
        bm25_start_time = time.time()
        retrieved_docs = self.retriever.retrieve(question, top_k)
        bm25_time = time.time() - bm25_start_time

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

        # 2. 计算BM25平均分数并选择模型
        avg_score = sum(score for _, score, _, _ in retrieved_docs) / len(
            retrieved_docs
        )
        selected_model = self.select_model_based_on_scores(avg_score)

        # 3. 构建上下文
        context = "\n\n".join(
            [
                f"Document {i+1}: {content}"
                for i, (_, _, content, _) in enumerate(retrieved_docs)
            ]
        )

        print(f"🤖 Generating answer with {selected_model}...")
        start_time = time.time()
        #answer = self.llm.generate_answer(selected_model, question, retrieved_docs)
        answer = "sss"
        llm_generation_time = time.time() - start_time

        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": [
                (doc_id, score, title) for doc_id, score, _, title in retrieved_docs
            ],
            "retrieval_count": len(retrieved_docs),
            "bm25_retrieval_time": bm25_time,
            "llm_generation_time": llm_generation_time,
            "avg_bm25_score": avg_score,
            "selected_model": selected_model,
        }


def main():
    """测试直接RAG Agent"""
    print("🚀 Testing Direct RAG Agent with 100 Wikipedia documents")
    print("=" * 60)

    # 初始化Agent
    agent = DirectRAGAgent(
        documents_json_path="wikipedia_documents.json", bm25_binary_path="/home/yimu3/approxMLIRTuner/examples/LLM_bm25/binary/bm25.exec"
    )

    # 从dataset JSON文件加载问题
    print("📖 Loading questions from enhanced_custom_dataset.json...")
    with open("enhanced_custom_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    test_questions = dataset["questions"]
    expected_answers = dataset["answers"]

    print(f"✅ Loaded {len(test_questions)} questions from dataset")

    print(f"\n🔍 Testing {len(test_questions)} questions:")

    # 统计正确率和存储所有结果
    correct_count = 0
    all_results = []

    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Question {i}/{len(test_questions)}: {question}")
        print(f"🎯 Expected Answer: {expected_answers[i-1]}")
        print("-" * 50)

        result = agent.answer_question(question, top_k=5)
        generated_answer = result["answer"]

        # 验证答案
        is_correct = check_answer_correct(generated_answer, expected_answers[i - 1])
        if is_correct:
            correct_count += 1
            print(f"✅ 答案正确!")
        else:
            print(f"❌ 答案不正确")

        print(f"🤖 Generated Answer: {generated_answer}")
        print(f"⏱️  BM25 Retrieval Time: {result['bm25_retrieval_time']:.3f} seconds")
        print(f"⏱️  LLM Generation Time: {result['llm_generation_time']:.2f} seconds")
        print(f"📊 Retrieved {result['retrieval_count']} documents")

        if result["retrieved_docs"]:
            print("📚 Retrieved documents:")
            for j, (doc_id, score, title) in enumerate(result["retrieved_docs"]):
                print(f"  {j+1}. {title} (Score: {score:.4f})")

        # 存储每个问题的详细结果
        question_result = {
            "question_id": i,
            "question": question,
            "expected_answers": expected_answers[i - 1],
            "generated_answer": generated_answer,
            "is_correct": is_correct,
            "bm25_retrieval_time": result["bm25_retrieval_time"],
            "llm_generation_time": result["llm_generation_time"],
            "retrieved_documents": [
                {"rank": j + 1, "doc_id": doc_id, "title": title, "score": score}
                for j, (doc_id, score, title) in enumerate(result["retrieved_docs"])
            ],
            "retrieval_count": result["retrieval_count"],
        }
        all_results.append(question_result)

    # 计算总体结果
    accuracy = correct_count / len(test_questions)
    total_bm25_time = sum(result["bm25_retrieval_time"] for result in all_results)
    total_llm_time = sum(result["llm_generation_time"] for result in all_results)
    avg_bm25_time = total_bm25_time / len(test_questions) if test_questions else 0
    avg_llm_time = total_llm_time / len(test_questions) if test_questions else 0

    # 创建完整的评估结果
    evaluation_results = {
        "metadata": {
            "total_questions": len(test_questions),
            "correct_answers": correct_count,
            "accuracy": accuracy,
            "accuracy_percentage": f"{accuracy:.1%}",
            "total_bm25_time": total_bm25_time,
            "average_bm25_time": avg_bm25_time,
            "total_llm_time": total_llm_time,
            "average_llm_time": avg_llm_time,
            "test_date": "2024-12-19",
            "model": "Gemma:2b",
            "retrieval_method": "BM25",
            "top_k": 3,
        },
        "questions": all_results,
    }

    # 保存结果到JSON文件
    output_file = "rag_evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

    # 打印总体结果
    total_time = total_bm25_time + total_llm_time
    print(f"\n📊 总体评估结果:")
    print(f"  总问题数: {len(test_questions)}")
    print(f"  正确答案数: {correct_count}")
    print(f"  准确率: {accuracy:.1%}")
    print(f"  总生成时间: {total_time:.3f} 秒")
    print(f"  总BM25检索时间: {total_bm25_time:.3f} 秒")
    print(f"  平均BM25检索时间: {avg_bm25_time:.3f} 秒")
    print(f"  总LLM生成时间: {total_llm_time:.2f} 秒")
    print(f"  平均LLM生成时间: {avg_llm_time:.2f} 秒")
    print(f"💾 详细结果已保存到: {output_file}")

    # 输出 tuner 需要的格式
    print(f"\nTUNER_OUTPUT: time={total_time:.6f} accuracy={accuracy:.6f}")


if __name__ == "__main__":
    main()
