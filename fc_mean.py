import numpy as np
from matplotlib import pyplot as plt

NUM_CLUSTER = 4
FUZZINESS_PARAMETER = 2.00
FEATURE_SIZE = 2
DATA_POINT_SIZE = 400


# X = np.random.uniform(0, 1000, (FEATURE_SIZE, DATA_POINT_SIZE))

def point_generator():
    x1 = np.random.uniform(0, 1, (FEATURE_SIZE, DATA_POINT_SIZE / 4))
    x1 = x1 * 4 + 10
    # print "X1: "
    # print x1
    x2 = np.random.uniform(0, 1, (FEATURE_SIZE, DATA_POINT_SIZE / 4))
    x2 = x2 * 4 - 10

    x3 = np.random.uniform(0, 1, (FEATURE_SIZE, DATA_POINT_SIZE / 4))
    x3 = x3 * -4 + 10
    x3[1:, ] = x3[1:, ] * -1

    x4 = np.random.uniform(0, 1, (FEATURE_SIZE, DATA_POINT_SIZE / 4))
    x4 = x4 * -4 - 10
    x4[1:, ] = x4[1:, ] * -1
    # print "X2: "
    # print x2
    x1 = np.append(x1, x2, axis=1)
    x2 = np.append(x3, x4, axis=1)
    x1 = np.append(x1, x2, axis=1)
    # print "new X1: "
    # print x1
    return x1


X = point_generator()
U = np.random.uniform(0, 1, (DATA_POINT_SIZE, NUM_CLUSTER))
U = U / U.sum(axis=1, keepdims=1)


def calculate_c_center():
    cluster_centers = np.zeros((FEATURE_SIZE, NUM_CLUSTER))
    for j in range(NUM_CLUSTER):
        numerate = denumerate = 0
        for i in range(DATA_POINT_SIZE):
            numerate = numerate + (U[i, j] ** FUZZINESS_PARAMETER) * X[:, i]
            denumerate = denumerate + U[i, j] ** FUZZINESS_PARAMETER
        cluster_centers[:, j] = numerate / denumerate

    return cluster_centers


def update_membership_value(cluster_centers):
    newU = np.zeros((DATA_POINT_SIZE, NUM_CLUSTER))
    for i in range(DATA_POINT_SIZE):
        for j in range(NUM_CLUSTER):
            numerate = destination(X[:, i], cluster_centers[:, j]) ** (-2 * (FUZZINESS_PARAMETER - 1))
            denumerate = 0
            for k in range(NUM_CLUSTER):
                denumerate = denumerate + destination(X[:, i], cluster_centers[:, k]) ** (-2 * (FUZZINESS_PARAMETER - 1))

            newU[i, j] = numerate / denumerate
    return newU


def destination(xk, vi):
    return np.linalg.norm(xk - vi)


def fcmean(num_iteration=100):
    cluster_centers = None
    while num_iteration > 0:
        num_iteration -= 1
        cluster_centers = calculate_c_center()
        U = update_membership_value(cluster_centers)
    label = np.argmax(U, axis=1)
    print(label)
    show_result(cluster_centers)


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
    fcmean(100)
