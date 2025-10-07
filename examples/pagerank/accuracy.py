def parse_pr_data(data_string):
    # Initialize an empty dictionary to store the results
    pr_values = {}
    
    # Split the input string into lines
    lines = data_string.strip().split('\n')
    
    # remove until the line starts with 'Time:'
    for i, line in enumerate(lines):
        if line.startswith('Time:'):
            lines = lines[i:]
            break
    
    # Extract the time from the first line
    execution_time = None
    if lines[0].startswith('Time:'):
        time_line = lines[0]
        execution_time = float(time_line.split(':')[1].split()[0])
    
    # Parse the pr values
    for line in lines:
        if line.startswith('pr('):
            # Extract the index and value
            parts = line.split('=')
            index_str = parts[0].strip()
            index = int(index_str[3:-1])  # Extract the number between 'pr(' and ')'
            value = float(parts[1].strip())
            
            # Store in dictionary
            pr_values[index] = value
    
    return execution_time, pr_values


def get_ranks(pr_values):
    """
    Return average ranks: higher value -> better (smaller) rank.
    Ties get the average of their rank positions.
    """
    items = sorted(pr_values.items(), key=lambda x: (-x[1], x[0]))  # value desc, then id asc
    ranks = {}
    i = 0
    n = len(items)
    while i < n:
        j = i + 1
        # group ties
        while j < n and items[j][1] == items[i][1]:
            j += 1
        # average rank of positions [i, j-1]
        avg_rank = (i + (j - 1)) / 2.0
        for k in range(i, j):
            node_id = items[k][0]
            ranks[node_id] = avg_rank
        i = j
    return ranks

def compute_similarity(pr_values1, pr_values2):
    """
    Rank-based similarity using Spearman footrule normalization.
    Returns a value in [0, 1], where 1 = identical ranking, 0 = worst-case (reverse order).
    """
    # 1) ranks: higher PR -> smaller rank index (0 is best)
    ranks1 = get_ranks(pr_values1)
    ranks2 = get_ranks(pr_values2)

    # 2) intersection
    common_keys = set(ranks1) & set(ranks2)
    N = len(common_keys)
    if N == 0:
        return 0.0
    if N == 1:
        return 1.0

    # 3) footrule distance
    rank_diff_sum = sum(abs(ranks1[k] - ranks2[k]) for k in common_keys)

    # 4) correct max (worst-case reverse ordering)
    if N % 2 == 0:
        max_diff = (N * N) / 2.0
    else:
        max_diff = (N * N - 1) / 2.0

    # 5) similarity in [0,1]
    similarity = 1.0 - (rank_diff_sum / max_diff)
    # numerical guard
    if similarity < 0.0:
        similarity = 0.0
    elif similarity > 1.0:
        similarity = 1.0

    return similarity


def get_gt(fname = "gt.txt"):
    with open(fname, "r") as f:
        s = f.read()
        time, gt = parse_pr_data(s)
    return gt

if __name__ == "__main__":
    get_gt()