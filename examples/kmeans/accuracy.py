import re
import math
import itertools

def parse_kernel_out(data_string):
    """
    Parses the k-means kernel's output string to extract computation time and
    the final centroid coordinates.

    Args:
        data_string: The stdout string from the kernel execution.

    Returns:
        A tuple containing (float: time_seconds, dict: centroids), where
        centroids is a dictionary mapping a cluster index to a list of its
        coordinate values.
    """
    time = -1.0
    centroids = {}

    # Regex to find the computation time in seconds.
    time_pattern = re.compile(r"K-means completed in ([\d.]+) seconds")
    
    # Regex to capture the centroid index and its coordinates.
    # It handles variable dimensions by finding all floating-point numbers in the line.
    centroid_pattern = re.compile(r"Centroid (\d+): \((.*?)\)")

    for line in data_string.splitlines():
        # Check for centroid lines
        centroid_match = centroid_pattern.search(line)
        if centroid_match:
            cluster_index = int(centroid_match.group(1))
            # Extract all floating-point numbers from the coordinate string
            coords_str = centroid_match.group(2)
            coords = [float(c) for c in re.findall(r"[\d.-]+", coords_str)]
            centroids[cluster_index] = coords
            continue

        # Check for the final time line
        time_match = time_pattern.search(line)
        if time_match:
            time = float(time_match.group(1))
            
    # The returned data structure 'D' is the dictionary of final centroids.
    return time, centroids


def compute_similarity(gt_out, approx_out):
    """
    Computes the accuracy between ground truth and approximated k-means outputs.

    Accuracy is based on the L2 norm of the final centroid coordinates. Since
    cluster labels can be permuted between runs, this function first finds the
    optimal matching between gt_out and approx_out centroids that minimizes
    the total Euclidean distance.

    Args:
        gt_out: The centroids dictionary from the exact kernel output.
        approx_out: The centroids dictionary from the approximated kernel output.

    Returns:
        A float representing the calculated accuracy.
    """
    if len(gt_out) != len(approx_out) or not gt_out:
        return 0.0

    # Sort the centroid keys to ensure consistent processing order
    gt_keys = sorted(gt_out.keys())
    approx_keys = sorted(approx_out.keys())

    # --- Find the best permutation of approximate centroids to match ground truth ---
    # This handles cases where cluster labels are swapped (e.g., gt cluster 0 matches approx cluster 3)
    best_mapping = {}
    min_total_dist = float('inf')

    # Iterate through all possible permutations of approximate cluster assignments
    for p in itertools.permutations(approx_keys):
        current_mapping = dict(zip(gt_keys, p))
        current_total_dist = 0
        for gt_key, approx_key in current_mapping.items():
            dist_sq = sum([(a - b)**2 for a, b in zip(gt_out[gt_key], approx_out[approx_key])])
            current_total_dist += math.sqrt(dist_sq)
        
        if current_total_dist < min_total_dist:
            min_total_dist = current_total_dist
            best_mapping = current_mapping

    # --- Construct flattened vectors based on the best matching ---
    y_exact = [coord for key in gt_keys for coord in gt_out[key]]
    
    # Use the best_mapping to order the approximate centroids correctly
    y_approx = [coord for key in gt_keys for coord in approx_out[best_mapping[key]]]

    # --- Calculate accuracy using the L2 norm formula ---
    diff_sum_sq = sum([(e - a)**2 for e, a in zip(y_exact, y_approx)])
    norm_diff = math.sqrt(diff_sum_sq)

    exact_sum_sq = sum([e**2 for e in y_exact])
    norm_exact = math.sqrt(exact_sum_sq)

    if norm_exact == 0.0:
        return 1.0 if norm_diff == 0.0 else 0.0
    
    accuracy = 1.0 - (norm_diff / norm_exact)
    
    return accuracy


def get_gt(fname="gt.txt"):
    """
    Reads the ground truth output from a file and parses it.
    """
    with open(fname, "r") as f:
        s = f.read()
        time, gt = parse_kernel_out(s)
    return gt

if __name__ == "__main__":
    # 1. Create a dummy ground truth file from the example output.
    gt_data_string = """
    K-means completed in 1.593 seconds
    Final centroids:
    Centroid 0: (82.54, 84.45)
    Centroid 1: (47.23, 15.19)
    Centroid 2: (45.17, 56.70)
    """
    with open("gt.txt", "w") as f:
        f.write(gt_data_string.strip())

    # 2. Get the parsed ground truth data.
    gt_centroids = get_gt()
    print(f"Ground Truth Centroids: {gt_centroids}")

    # 3. Create a dummy approximate output. Note that Centroid 0 and 1 are swapped
    #    and have slight errors compared to the ground truth.
    approx_data_string = """
    K-means completed in 0.850 seconds
    Final centroids:
    Centroid 0: (47.30, 15.25)
    Centroid 1: (82.60, 84.50)
    Centroid 2: (45.10, 56.80)
    """
    
    # 4. Parse the approximate output and compute similarity.
    approx_time, approx_centroids = parse_kernel_out(approx_data_string)
    print(f"Approximate Centroids: {approx_centroids}")
    print(f"Approximate Time: {approx_time} s")
    
    accuracy = compute_similarity(gt_centroids, approx_centroids)
    print(f"\nCalculated Accuracy: {accuracy:.6f}")