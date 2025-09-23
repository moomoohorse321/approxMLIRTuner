import re
import math

def parse_kernel_out(data_string):
    """
    Parses the kernel's string output to extract the computation time and a
    data structure containing the core data needed for accuracy calculations.

    Args:
        data_string: The stdout string from the kernel execution.

    Returns:
        A tuple containing (float: time_ms, dict: scores), where scores is a
        dictionary mapping document index to its BM25 score.
    """
    time = -1.0
    scores = {}

    # Regex to find the computation time in milliseconds.
    time_pattern = re.compile(r"Computation time: ([\d.]+) ms")
    
    # Regex to find the score for each document.
    # It captures the document index (int) and its score (float).
    score_pattern = re.compile(r"Rank \d+: Doc (\d+) \(Score: ([\d.-]+)\)")

    for line in data_string.splitlines():
        # Check for score lines
        score_match = score_pattern.search(line)
        if score_match:
            doc_index = int(score_match.group(1))
            score = float(score_match.group(2))
            scores[doc_index] = score
            continue

        # Check for the final time line
        time_match = time_pattern.search(line)
        if time_match:
            time = float(time_match.group(1))
            
    # The returned data structure 'D' is a dictionary of scores.
    return time, scores


def compute_similarity(gt_out, approx_out):
    """
    Computes the accuracy between the ground truth and approximated outputs.
    
    For the BM25 benchmark, accuracy is defined by the L2 norm error as specified
    in the approxMLIR paper.
    The formula is: Accuracy = 1 - (||Y_exact - Y_approx||_2 / ||Y_exact||_2)

    Args:
        gt_out: The scores dictionary from the exact kernel output.
        approx_out: The scores dictionary from the approximated kernel output.

    Returns:
        A float representing the calculated accuracy.
    """
    # Ensure both dictionaries have the same keys (doc indices) and are sorted.
    if gt_out.keys() != approx_out.keys():
        print("Warning: Mismatch in document indices between outputs.")
        return 0.0

    sorted_indices = sorted(gt_out.keys())
    
    y_exact = [gt_out[i] for i in sorted_indices]
    y_approx = [approx_out[i] for i in sorted_indices]
    
    # Calculate the L2 norm of the difference vector (Y_exact - Y_approx).
    diff_sum_sq = sum([(e - a)**2 for e, a in zip(y_exact, y_approx)])
    norm_diff = math.sqrt(diff_sum_sq)

    # Calculate the L2 norm of the exact vector (Y_exact).
    exact_sum_sq = sum([e**2 for e in y_exact])
    norm_exact = math.sqrt(exact_sum_sq)
    # print(y_exact)
    # Handle the edge case where the exact output vector is all zeros.
    if norm_exact == 0.0:
        # If the approximate output is also all zeros, they are identical (perfect accuracy).
        # Otherwise, any difference results in zero accuracy.
        return 1.0 if norm_diff == 0.0 else 0.0
    
    accuracy = 1.0 - (norm_diff / norm_exact)
    
    return accuracy


def get_gt(fname = "gt.txt"):
    """
    Reads the ground truth output from a file and parses it.
    """
    with open(fname, "r") as f:
        s = f.read()
        time, gt = parse_kernel_out(s)
    return gt

if __name__ == "__main__":
    # Example usage:
    # 1. Create a dummy ground truth file from the example output.
    gt_data_string = """
    Query: "hello"

    Ranking documents:
    Rank 1: Doc 0 (Score: 0.0000) - "information normalize embedding tfidf frequency index document over brown probabilistic weight ranking engine field model pagerank."
    Rank 2: Doc 1 (Score: 0.0000) - "text quality quality quality information text index measure measure search the brown engine lazy text pagerank."
    Rank 3: Doc 2 (Score: 0.0000) - "probabilistic walk measure quality approximate brown language engine pagerank jumps relevance."
    Rank 4: Doc 3 (Score: 0.0000) - "modeling approximate over text document measure space retrieval jumps dog embedding bm25 weight random quick pagerank."
    Rank 5: Doc 4 (Score: 0.0000) - "term walk search ranking document vector search."
    Rank 6: Doc 5 (Score: 0.0000) - "quick modeling random over random vector space information field modeling search retrieval jumps model."
    Rank 7: Doc 6 (Score: 0.0000) - "ranking field lazy approximate probabilistic weight text analysis over lazy inverse document."
    Rank 8: Doc 7 (Score: 0.0000) - "space fox frequency tfidf measure graph embedding."
    Rank 9: Doc 8 (Score: 0.0000) - "document document search vector walk query."
    Rank 10: Doc 9 (Score: 0.0000) - "language system length field the document tfidf brown brown probabilistic text search search."

    Computation time: 0.015 ms
    """
    with open("gt.txt", "w") as f:
        f.write(gt_data_string.strip())

    # 2. Get the parsed ground truth data.
    gt_scores = get_gt()
    print(f"Ground Truth Scores: {gt_scores}")

    # 3. Create a dummy approximate output with some errors.
    approx_data_string = """
    Query: "hello"

    Ranking documents:
    Rank 1: Doc 0 (Score: 0.1000) 
    Rank 2: Doc 1 (Score: -0.0500)
    Rank 3: Doc 2 (Score: 0.0000)
    Rank 4: Doc 3 (Score: 0.0000)
    Rank 5: Doc 4 (Score: 0.2000)
    Rank 6: Doc 5 (Score: 0.0000)
    Rank 7: Doc 6 (Score: -0.1500)
    Rank 8: Doc 7 (Score: 0.0000)
    Rank 9: Doc 8 (Score: 0.0000)
    Rank 10: Doc 9 (Score: 0.0500)

    Computation time: 0.008 ms
    """
    
    # 4. Parse the approximate output and compute similarity.
    approx_time, approx_scores = parse_kernel_out(approx_data_string)
    print(f"Approximate Scores: {approx_scores}")
    print(f"Approximate Time: {approx_time} ms")
    
    # In this specific case, the exact output is all zeros.
    # This will trigger the edge case in compute_similarity.
    accuracy = compute_similarity(gt_scores, approx_scores)
    print(f"\nCalculated Accuracy: {accuracy:.4f}")