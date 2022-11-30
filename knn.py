from cmath import inf
import numpy as np
import random

# Distance Computation
def compute_distance(matrix, second_matrix, i, j):
    # Lose tiny accuracy while increase speed
    dis = np.sqrt(np.sum(np.square(matrix[i] - second_matrix[j])))
    """
    The following coding compute the exact distance, but much slower
    len_tm = len(tm)
    for i in range(len_tm):
        if i in empty_idx_i:
            continue 
        else:
            dis = dis + (tm[i] - tn[i]) * (tm[i] - tn[i])
    dis = np.sqrt(dis)
    """

    return np.sqrt(dis)


# Find missing idx
def find_zero_idx(tx):
    res = []
    for i in range(len(tx)):
        if tx[i] == -999:
            res.append(i)
    return res


# Using KNN to fill in the missing data (Accurate but slow)
def knn_auto_fill(
    matrix, normallized_matrix, train_matrix=None, normalized_train_matrix=None
):
    if normalized_train_matrix is None:
        normalized_train_matrix = normallized_matrix
        train_matrix = matrix
    # The missing idx for each training data

    empty_idx = [False if -999 not in line else find_zero_idx(line) for line in matrix]
    empty_idx_train = [
        False if -999 not in line else find_zero_idx(line) for line in train_matrix
    ]

    # Get the idx that has no missing values
    full_idx_list = []
    for i in range(len(empty_idx_train)):
        if not empty_idx[i]:
            full_idx_list.append(i)

    for i in range(len(matrix)):
        if i % 50 == 0:
            print(i, len(matrix) - i)
        empty_idx_i = empty_idx[i]
        best_distance = float("inf")
        best_idx = i
        if not empty_idx_i:
            continue
        for j in full_idx_list:
            # Compute the Best distances
            cur_distance = compute_distance(
                normallized_matrix, normalized_train_matrix, i, j
            )
            if cur_distance < best_distance:
                best_distance = cur_distance
                best_idx = j
        for k in empty_idx_i:
            matrix[i][k] = train_matrix[best_idx][k]
    print(matrix)
    return matrix


# Using Monte-Carlo-based KNN to fill in the missing data (Faster but use sampling)
def knn_random_fill(
    matrix, normallized_matrix, train_matrix=None, normalized_train_matrix=None
):
    if normalized_train_matrix is None:
        normalized_train_matrix = normallized_matrix
        train_matrix = matrix
    # The missing idx for each training data

    empty_idx = [False if -999 not in line else find_zero_idx(line) for line in matrix]
    empty_idx_train = [
        False if -999 not in line else find_zero_idx(line) for line in train_matrix
    ]

    # Get the idx that has no missing values
    full_idx_list = []
    for i in range(len(empty_idx_train)):
        if not empty_idx[i]:
            full_idx_list.append(i)

    size_full_idx = len(full_idx_list)
    total_sample_cnt = int(size_full_idx * 0.2)
    for i in range(len(matrix)):
        if i % 50 == 0:
            print(i, len(matrix) - i)
        empty_idx_i = empty_idx[i]
        best_distance = float("inf")
        best_idx = i
        if not empty_idx_i:
            continue

        sample_cnt = total_sample_cnt
        while sample_cnt >= 0:
            # Compute the Best distances
            rand_idx = random.randint(0, size_full_idx - 2)
            j = full_idx_list[rand_idx]
            cur_distance = compute_distance(
                normallized_matrix, normalized_train_matrix, i, j
            )

            if cur_distance < best_distance:
                best_distance = cur_distance
                best_idx = j

            sample_cnt = sample_cnt - 1
        for k in empty_idx_i:
            matrix[i][k] = train_matrix[best_idx][k]
    return matrix


# test function for KNN
def test_knn():
    test_list = [
        [-999, 4, 5, 8],
        [9, 7, 7, 4],
        [8, 6, -999, 2],
        [3, 4, 5, 7],
        [6, 4, 8, 5],
    ]
    test_np = np.array(test_list)
    print(test_np)
    test_np = knn_auto_fill(test_np, test_np)
    print(test_np)


if __name__ == "main":
    test_knn()
