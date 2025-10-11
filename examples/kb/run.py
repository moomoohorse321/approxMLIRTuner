from get_embedding import query_embedding
import subprocess

# exec_path = "/home/hao/Polygeist/opentuner/examples/kb/binary/kb.exec"
exec_path  = "/home/hao/Polygeist/tools/cgeist/Test/approxMLIR/kb"

cmd = [
    exec_path,
    query_embedding,
    "20"
]

with open("input.txt", "r") as f:
    doc_input = f.read()

run_result = subprocess.run(
    cmd,
    input=doc_input,
    capture_output=True,
    text=True,
    timeout=120,
    encoding="utf-8",
)

print(run_result.stdout)