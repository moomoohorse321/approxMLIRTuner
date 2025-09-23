import re
import math
import numpy as np

def parse_kernel_out(stdout_string, file_content_string):
    """
    Parses the Black-Scholes kernel's outputs to extract computation time
    from stdout and the final option prices from the output file.

    Args:
        stdout_string: The string captured from the kernel's standard output.
        file_content_string: The string content from the kernel's output data file.

    Returns:
        A tuple containing (float: time_seconds, list: prices). The prices list
        is the core data structure 'D' for accuracy calculation.
    """
    time = -1.0
    prices = []

    # --- Parse Time from Standard Output ---
    time_pattern = re.compile(r"Elapsed: ([\d.]+)")
    time_match = time_pattern.search(stdout_string)
    if time_match:
        time = float(time_match.group(1))

    # --- Parse Prices from File Content ---
    lines = file_content_string.strip().splitlines()
    for line in lines:
        # Skip header/comment lines for price data
        if line.startswith("#"):
            continue
        
        # The price is the last value in each space-separated row.
        # Check for 8 columns to ensure it's a data line.
        parts = line.split()
        if len(parts) == 8:
            try:
                price = float(parts[-1])
                prices.append(price)
            except (ValueError, IndexError):
                # Ignore lines that don't conform to the expected format
                continue
                
    return time, prices


def compute_similarity(gt_out, approx_out):
    """
    Computes the accuracy between the ground truth and approximated outputs
    for the Black-Scholes benchmark.
    
    As specified in the paper, accuracy is defined by the L1 norm error:
    Accuracy = 1 - (||Y_exact - Y_approx||_1 / ||Y_exact||_1)

    Args:
        gt_out: A list of option prices from the exact kernel output.
        approx_out: A list of option prices from the approximated kernel output.

    Returns:
        A float representing the calculated accuracy.
    """
    if len(gt_out) != len(approx_out) or not gt_out:
        return 0.0

    # Convert lists to numpy arrays for efficient vector operations
    y_exact = np.array(gt_out)
    y_approx = np.array(approx_out)
    
    # Calculate the L1 norm of the difference vector (error)
    norm_diff = np.linalg.norm(y_exact - y_approx, ord=1)

    # Calculate the L1 norm of the exact vector
    norm_exact = np.linalg.norm(y_exact, ord=1)
    
    # Handle the edge case where the exact output vector is all zeros
    if norm_exact == 0.0:
        # If the approximate output is also all zeros, they are identical (perfect accuracy).
        # Otherwise, any difference results in zero accuracy.
        return 1.0 if norm_diff == 0.0 else 0.0
    
    accuracy = 1.0 - (norm_diff / norm_exact)
    
    return accuracy


def get_gt(fname = "gt.txt"):
    """
    Reads the ground truth price data from a file. This helper is used to
    get the data for accuracy comparison.
    """
    with open(fname, "r") as f:
        file_content = f.read()
    
    # We only need the price data for the ground truth comparison,
    # so we can pass an empty string for the stdout part.
    _, gt_prices = parse_kernel_out("", file_content)
    return gt_prices

if __name__ == "__main__":
    # 1. Create dummy stdout and file content strings, as they would be
    #    captured from a program run.
    gt_stdout_string = """
PARSEC Benchmark Suite
Elapsed: 0.000012
--- Black-Scholes Results ---
Number of options: 5
    """
    
    gt_file_content_string = """
# idx otype sptprice strike rate volatility otime price
0 0 104.04697420979385 103.20570280398091 0.09837464949334096 0.66036227763133859 0.25861171233945907 11.995828196031709
1 0 90.001927104165347 66.14613716846435 0.073517607777189353 0.29628741975348916 0.21252998499178233 0.031155870246603157
2 1 98.634923771534375 78.9782576824866 0.058589831311977141 0.38817369783190475 0.90350051381521457 4.1491194961006244
3 1 68.562395270335614 115.69742280944126 0.06444915804330642 0.79510997790055749 2.4218865100874134 55.538214207500403
4 1 53.003473070196769 111.68679431326099 0.094699998765607349 0.52582771941494688 2.7163769759685761 43.340418706682449
    """
    # Simulate writing the ground truth data file for get_gt() to read
    with open("gt.txt", "w") as f:
        f.write(gt_file_content_string.strip())

    # 2. Parse the ground truth outputs separately.
    gt_time, gt_prices = parse_kernel_out(gt_stdout_string, gt_file_content_string)
    print(f"Ground Truth Time: {gt_time} s")
    print(f"Ground Truth Prices (first 5): {gt_prices}")

    # 3. Create a dummy approximate output with some errors.
    approx_prices = [
        12.0,      # Small error
        0.0,       # Large relative error
        4.2,       # Small error
        55.0,      # Larger error
        43.340418  # Very small error
    ]
    
    print(f"Approximate Prices (first 5): {approx_prices}")
    
    # 4. Compute the similarity/accuracy.
    accuracy = compute_similarity(gt_prices, approx_prices)
    print(f"\nCalculated Accuracy: {accuracy:.6f}")

