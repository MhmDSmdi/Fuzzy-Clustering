import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

NUM_CLUSTER = 4
FUZZINESS_PARAMETER = 2.00
FEATURE_SIZE = 2
DATA_POINT_SIZE = 1000


def point_generator():
    x1 = np.random.uniform(0, 1, (FEATURE_SIZE, DATA_POINT_SIZE / 4))
    x1 = x1 * 250 + 10
    x2 = np.random.uniform(0, 1, (FEATURE_SIZE, DATA_POINT_SIZE / 4))
    x2 = x2 * 250 - 10
    x3 = np.random.uniform(0, 1, (FEATURE_SIZE, DATA_POINT_SIZE / 4))
    x3 = x3 * -250 + 10
    x3[1:, ] = x3[1:, ] * -1
    x4 = np.random.uniform(0, 1, (FEATURE_SIZE, DATA_POINT_SIZE / 4))
    x4 = x4 * -250 - 10
    x4[1:, ] = x4[1:, ] * -1
    x1 = np.append(x1, x2, axis=1)
    x2 = np.append(x3, x4, axis=1)
    x1 = np.append(x1, x2, axis=1)
    return x1


def calculate_c_center(U, X):
    cluster_centers = np.zeros((FEATURE_SIZE, NUM_CLUSTER))
    for j in range(NUM_CLUSTER):
        numerate = denumerate = 0
        for i in range(DATA_POINT_SIZE):
            numerate = numerate + (U[i, j] ** FUZZINESS_PARAMETER) * X[:, i]
            denumerate = denumerate + U[i, j] ** FUZZINESS_PARAMETER
        cluster_centers[:, j] = numerate / denumerate

    return cluster_centers


def update_membership_value(X, cluster_centers):
    newU = np.zeros((DATA_POINT_SIZE, NUM_CLUSTER))
    for i in range(DATA_POINT_SIZE):
        for j in range(NUM_CLUSTER):
            numerate = destination(X[:, i], cluster_centers[:, j]) ** (2 / (1 - FUZZINESS_PARAMETER))
            denominator = 0
            for k in range(NUM_CLUSTER):
                denominator = denominator + destination(X[:, i], cluster_centers[:, k]) ** (
                        2 / (1 - FUZZINESS_PARAMETER))
            newU[i, j] = numerate / denominator
    return newU


def destination(xk, vi):
    return np.linalg.norm(xk - vi)


def fcmean(num_iteration=100):
    X = point_generator()
    U = np.random.uniform(0, 1, (DATA_POINT_SIZE, NUM_CLUSTER))
    U = U / U.sum(axis=0, keepdims=1)
    cluster_centers = None
    i = num_iteration
    while i > 0:
        i -= 1
        cluster_centers = calculate_c_center(U, X)
        U = update_membership_value(X, cluster_centers)
    labels = np.argmax(U, axis=1)
    print(labels)
    show_result(X, labels, cluster_centers, num_iteration)


def show_result(X, labels, cluster_centers, num_iteration):
    x_data = X[0, :]
    y_data = X[1, :]
    x_centroid = cluster_centers[0, :]
    y_centroid = cluster_centers[1, :]
    txt = "NumCluster = {}, NumIteration = {}, NumData = {}, FuzzyParameter = {}".format(NUM_CLUSTER, num_iteration, DATA_POINT_SIZE, FUZZINESS_PARAMETER)
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(0, NUM_CLUSTER, NUM_CLUSTER + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    scat = ax.scatter(x_data, y_data, s=30, c=labels, cmap=cmap, norm=norm)
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Clusters')
    ax.set_title("FCM Algorithm\n" + txt)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    plt.scatter(x_centroid, y_centroid, s=400, marker='+', c="red")
    plt.show()


if __name__ == '__main__':
    fcmean(100)
