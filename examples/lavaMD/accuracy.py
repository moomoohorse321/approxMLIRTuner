import os
import struct
import math
import re

def _read_size_t(f):
    pos = f.tell()
    raw = f.read(8)
    if len(raw) == 8:
        (n,) = struct.unpack("Q", raw)
        # 简单 sanity：过于离谱则回退 4B
        if n < 10_000_000_000:
            return n
    f.seek(pos)
    raw = f.read(4)
    if len(raw) != 4:
        raise RuntimeError("Failed to read size_t header")
    (n4,) = struct.unpack("I", raw)
    return n4

def _read_result_bin(path):
    with open(path, "rb") as f:
        N = _read_size_t(f)
        rec_size = 8 * 4  # FOUR_VECTOR: 4*double
        blob = f.read(N * rec_size)
        if len(blob) < N * rec_size:
            raise RuntimeError("result file truncated")
    V = [0.0] * N
    F = [0.0] * (3 * N)
    off = 0
    for i in range(N):
        v, x, y, z = struct.unpack_from("dddd", blob, off)
        V[i] = v
        j = 3 * i
        F[j] = x; F[j+1] = y; F[j+2] = z
        off += rec_size
    return {"V": V, "F": F}

def _l2(vec):
    return math.sqrt(sum(x*x for x in vec))

def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def parse_pr_data(data_string):
    """
        It took from result.txt
    """
    execution_time = None

    if data_string:
        lines = data_string.splitlines()
            
        for line in lines:
            s = line.strip()
            m = re.compile(r"^\s*Total execution time:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?:ms)?\s*$").match(s)
            if m:
                execution_time = float(m.group(1))
                break

    result_path = os.environ.get("RESULT_FILE", "result.txt")
    if not os.path.isfile(result_path):
        return execution_time, {"V": [], "F": []}

    lavamd_values = _read_result_bin(result_path)
    return execution_time, lavamd_values


def get_gt(fname="gt.txt"):
    return _read_result_bin(fname)


def compute_similarity(gt_values, cand_values):
    # 基本健壮性
    if not isinstance(gt_values, dict) or not isinstance(cand_values, dict):
        return 0.0
    if "V" not in gt_values or "F" not in gt_values:
        return 0.0
    if "V" not in cand_values or "F" not in cand_values:
        return 0.0

    Vb, Fb = gt_values["V"], gt_values["F"]
    Va, Fa = cand_values["V"], cand_values["F"]

    # 尺寸检查
    if len(Va) == 0 and len(Vb) == 0:
        return 1.0
    if len(Va) != len(Vb) or len(Fa) != len(Fb):
        return 0.0

    eps = 1e-12
    # 相对 L2 误差
    diffV = [a - b for a, b in zip(Va, Vb)]
    diffF = [a - b for a, b in zip(Fa, Fb)]
    relL2_V = _l2(diffV) / max(_l2(Vb), eps)
    relL2_F = _l2(diffF) / max(_l2(Fb), eps)

    # 力向量方向的余弦相似度
    denom = max(_l2(Fa) * _l2(Fb), eps)
    cos_F = _dot(Fa, Fb) / denom

    # 将误差映射到 [0,1]：exp 衰减更平滑；与 cos 融合
    base_sim = math.exp(-(relL2_V + relL2_F))            # 误差越小越接近 1
    cos_term = (cos_F + 1.0) / 2.0                        # [-1,1] -> [0,1]
    sim = 0.5 * base_sim + 0.5 * max(0.0, min(1.0, cos_term))

    return max(0.0, min(1.0, sim))