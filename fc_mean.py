import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

FUZZINESS_PARAMETER = 2.00


def point_generator(num_cluster, feature_size, num_data):
    data = np.zeros((feature_size, 0))
    cof = 1
    # for i in range(num_cluster):

    # x1 = np.random.beta(a, b, (feature_size, num_data / num_cluster)) * 2000
    # x1[0, :] = np.random.choice([1, -1, 2, -2]) * x1[0, :] + np.random.randint(-1000, 1000)
    # x1[1, :] = np.random.choice([1, -1, 2, -2]) * x1[1, :] + np.random.randint(-1000, 1000)
    # cof += 1
    # data = np.append(x1, data, axis=1)

    a = np.random.randint(10, 15)
    b = np.random.randint(8, 12)
    x1 = np.random.beta(a, b, (feature_size, num_data / num_cluster)) * 500
    x1[0, :] = 2 * x1[0, :] + 300
    x1[1, :] = x1[1, :]

    x2 = np.random.beta(a, b, (feature_size, num_data / num_cluster)) * 500
    x2[0, :] = 2 * x2[0, :]
    x2[1, :] = x2[1, :] + 250

    x3 = np.random.beta(a, b, (feature_size, num_data / num_cluster)) * 500
    x3[0, :] = 2 * x3[0, :] - 300
    x3[1, :] = x3[1, :]

    data = np.append(x1, x2, axis=1)
    data = np.append(data, x3, axis=1)
    return data


def calculate_c_center(U, X):
    cluster_centers = np.zeros((X.shape[0], U.shape[1]))
    for j in range(U.shape[1]):
        numerate = denumerate = 0
        for i in range(X.shape[1]):
            numerate = numerate + (U[i, j] ** FUZZINESS_PARAMETER) * X[:, i]
            denumerate = denumerate + U[i, j] ** FUZZINESS_PARAMETER
        cluster_centers[:, j] = numerate / denumerate

    return cluster_centers


def update_membership_value(X, cluster_centers):
    newU = np.zeros((X.shape[1], cluster_centers.shape[1]))
    for i in range(X.shape[1]):
        for j in range(cluster_centers.shape[1]):
            numerate = destination(X[:, i], cluster_centers[:, j]) ** (2 / (1 - FUZZINESS_PARAMETER))
            denominator = 0
            for k in range(cluster_centers.shape[1]):
                denominator = denominator + destination(X[:, i], cluster_centers[:, k]) ** (
                        2 / (1 - FUZZINESS_PARAMETER))
            newU[i, j] = numerate / denominator
    return newU


def destination(xk, vi):
    return np.linalg.norm(xk - vi)


def fcmean(data, num_clusters=3, num_iteration=100):
    U = np.random.uniform(0, 1, (data.shape[1], num_clusters))
    U = U / U.sum(axis=0, keepdims=1)
    cluster_centers = labels = None
    i = 0
    while i != num_iteration:
        cluster_centers = calculate_c_center(U, data)
        U = update_membership_value(data, cluster_centers)
        labels = np.argmax(U, axis=1)
        if i in [2, 5, 10, 20, 100]:
            show_result(data, labels, cluster_centers, i)
        i += 1
    print(labels.shape)
    show_result(data, labels, cluster_centers, num_iteration)


def show_result(X, labels, cluster_centers, num_iteration):
    x_data = X[0, :]
    y_data = X[1, :]
    x_centroid = cluster_centers[0, :]
    y_centroid = cluster_centers[1, :]
    txt = "(Cluster = {}, Iteration = {}, Data = {}, FuzzyParameter = {})".format(cluster_centers.shape[1],
                                                                                  num_iteration, X.shape[1],
                                                                                  FUZZINESS_PARAMETER)
    # cmap = plt.cm.jet
    # cmaplist = [cmap(i) for i in range(cmap.N)]
    # cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    # bounds = np.linspace(0, cluster_centers.shape[1], cluster_centers.shape[1] + 1)
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # scat = ax.scatter(x_data, y_data, s=30, c=labels)#, cmap=cmap, norm=norm)
    # cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    # cb.set_label('Clusters')
    for i in range(len(labels)):
        label = labels[i]
        if label == 0:
            plt.plot(x_data[i], y_data[i], 'ob')
        if label == 1:
            plt.plot(x_data[i], y_data[i], 'or')
        if label == 2:
            plt.plot(x_data[i], y_data[i], 'og')
    # ax.set_title("FCM Algorithm\n" + txt)
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    plt.ylabel("Y-axis")
    plt.xlabel("X-axis")
    plt.title("FCM Algorithm\n" + txt)
    plt.scatter(x_centroid, y_centroid, s=400, marker='+', c="yellow")
    plt.show()


def show_input_data(X):
    x_data = X[0, :]
    y_data = X[1, :]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title("FCM Algorithm\n Input Data")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    scat = ax.scatter(x_data, y_data, s=30)
    plt.show()


if __name__ == '__main__':
    num_cluster = 3
    feature_size = 2
    num_data = 1500
    X = point_generator(num_cluster=num_cluster, feature_size=feature_size, num_data=num_data)
    show_input_data(X)
    fcmean(X)
