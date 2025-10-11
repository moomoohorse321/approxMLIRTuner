import re
import math

def parse_kernel_out(data_string):
    """
    Parses the kernel's string output to extract the computation time and the
    ranked list of documents.

    Args:
        data_string (str): The multiline string output from the kernel.

    Returns:
        tuple: A tuple containing:
            - time (float): The execution time in milliseconds. Returns 0.0 if not found.
            - data (list): A list of tuples, where each tuple represents a
                           ranked document as (rank, doc_id, score).
    """
    # --- Time Parsing ---
    # Correctly look for "Elapsed 1807.451 ms" at the start of the string.
    time_match = re.search(r"^\s*Elapsed\s+([\d\.]+)\s*ms", data_string)
    time = float(time_match.group(1)) if time_match else 0.0

    # --- Data Parsing ---
    # Regex to capture rank, doc ID, and score from each line.
    pattern = re.compile(r"Rank\s+(\d+):\s+Doc\s+(\S+)\s+\(Score:\s+([\d\.]+)\)")
    
    parsed_data = []
    for line in data_string.strip().split('\n'):
        match = pattern.match(line)
        if match:
            rank = int(match.group(1))
            doc_id = match.group(2)
            score = float(match.group(3))
            parsed_data.append((rank, doc_id, score))
            
    return time, parsed_data

def get_rbo_result(gt_string, approx_string, p=0.9):
    """
    Calculates a rescaled RBO score that is properly normalized to 1 for
    finite lists of the same length.
    """
    _, gt_data = parse_kernel_out(gt_string)
    approx_time, approx_data = parse_kernel_out(approx_string)
    
    gt_list = [doc_id for _, doc_id, _ in gt_data]
    approx_list = [doc_id for _, doc_id, _ in approx_data]

    if not gt_list or not approx_list:
        return approx_time, 0.0

    # --- Standard RBO Calculation (as before) ---
    max_depth = max(len(gt_list), len(approx_list))
    rbo_score_raw = 0.0
    gt_set, approx_set = set(), set()
    
    for d in range(1, max_depth + 1):
        if d <= len(gt_list):
            gt_set.add(gt_list[d-1])
        if d <= len(approx_list):
            approx_set.add(approx_list[d-1])
        
        overlap = len(gt_set.intersection(approx_set))
        agreement_at_d = overlap / d
        rbo_score_raw += (p**(d - 1)) * agreement_at_d
    
    unnormalized_score = (1 - p) * rbo_score_raw

    # --- Rescaling Factor ---
    # Calculate the maximum possible score for a list of this depth
    max_possible_score_raw = 0.0
    for d in range(1, max_depth + 1):
        max_possible_score_raw += (p**(d-1)) # Agreement is always 1 for identical lists
    
    max_possible_score = (1-p) * max_possible_score_raw

    # --- Return the Rescaled Score ---
    if max_possible_score == 0:
        return approx_time, 0.0
    
    rescaled_accuracy = unnormalized_score / max_possible_score
    return approx_time, rescaled_accuracy

def get_gt(fname = "gt.txt"):
    """
    Reads the ground truth output from a file and parses it.
    """
    with open(fname, "r", errors="ignore") as f:
        s = f.read()
    return s

if __name__ == "__main__":
    # Ground truth output string, using the format you provided
    gt_output_string = """
    Elapsed 1807.451 ms
    Rank 1: Doc 14554 (Score: 0.6887) - "Moon landing"
    Rank 2: Doc 4 (Score: 0.6395) - "Apollo 11"
    Rank 3: Doc 3201 (Score: 0.5819) - "List of missions to the Moon"
    Rank 4: Doc 47313 (Score: 0.5594) - "List of astronauts by first flight"
    Rank 5: Doc 43478 (Score: 0.5496) - "List of Apollo astronauts"
    """

    # Approximated output, with a different time and modified results
    approx_output_string = """
    Elapsed 950.25 ms
    Rank 1: Doc 14554 (Score: 0.65) - "Moon landing"
    Rank 2: Doc 4 (Score: 0.6395) - "List of missions to the Moon"
    Rank 3: Doc 3201 (Score: 0.5894) - "Apollo 11"
    Rank 4: Doc 43478 (Score: 0.5594) - "List of astronauts by first flight"
    Rank 5: Doc 47313 (Score: 0.5496) - "List of Apollo astronauts"
    """

    # Calculate results using both methods
    time_rbo, accuracy_rbo = get_rbo_result(gt_output_string, approx_output_string, p=0.95)


    print("--- RBO Result ---")
    print(f"Execution Time: {time_rbo} ms")
    print(f"Accuracy (p=0.95): {accuracy_rbo:.4f}")