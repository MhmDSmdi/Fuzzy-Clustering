import math
from matplotlib import pyplot as plt
import numpy as np
import operator

FEATURE_SIZE = 4
DATA_POINT_SIZE = 5
X = np.random.uniform(0, 1, (DATA_POINT_SIZE, FEATURE_SIZE))
FUZZY_PARAMETER = 2.00


def cal_cluster_center(data, num_cluster):
    cluster_mem_val = zip(*data)
    cluster_centers = list()
    for j in range(num_cluster):
        x = list(cluster_mem_val[j])
        xraised = [mu ** FUZZY_PARAMETER for mu in x]
        denominator = sum(xraised)
        temp_num = list()
        for i in range(DATA_POINT_SIZE):
            data_point = list(data[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z / denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers


def update_partition_matrix(data, cluster_center, num_cluster):
    p = float(2 / (FUZZY_PARAMETER - 1))
    for i in range(DATA_POINT_SIZE):
        x = list(data[i])
        distances = [np.linalg.norm(map(operator.sub, x, cluster_center[j])) for j in range(num_cluster)]
        for j in range(num_cluster):
            den = sum([math.pow(float(distances[j] / distances[c]), p) for c in range(num_cluster)])
            data[i][j] = float(1 / den)
    return data


def find_labels(data):
    cluster_labels = list()
    for i in range(DATA_POINT_SIZE):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(data[i]))
        cluster_labels.append(idx)
    return cluster_labels


def fcmean(data=X, num_cluster=3, num_iteraton=10000):
    if num_cluster < 2 or data.shape[1] < num_cluster:
        raise Exception("Your input is not valid")
    else:
        for i in range(num_iteraton):
            cluster_centers = cal_cluster_center(data, num_cluster)
            data = update_partition_matrix(data, cluster_centers, num_cluster)
            labels = find_labels(data)
    return labels


def show_result(cluster_centers):
    x_data = X[0, :]
    y_data = X[1, :]
    x_centroid = cluster_centers[0, :]
    y_centroid = cluster_centers[1, :]
    plt.title("Result of FSM Algorithm")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x_data, y_data, "ob")
    plt.plot(x_centroid, y_centroid, "or")
    plt.show()


if __name__ == '__main__':
    a = fcmean(num_cluster=2)
    print a
