import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn import cluster
except ImportError:
    print("Warning: scikit-learn not installed")
    print("pip install scikit-learn")
    pass


def normalize_features(features):
    """Divide each column by its max.

    Returns the maxes as well to denormalize later
    """
    maxes = np.max(features, axis=0)
    return features / maxes, maxes


def cluster_blobs(blobs):
    print("Taking abs value of blobs")
    blobs = np.abs(blobs)
    X_size_mag = blobs[:, [2, 3]]
    print("Pre normalizing:", np.max(X_size_mag, axis=0))
    # X_size_mag[:, 1] = X_size_mag[:, 1]**3
    # X_size_mag[:, 0] = X_size_mag[:, 0]**(1 / 3)
    X_size_mag, maxes = normalize_features(X_size_mag)
    print("Post normalizing: ", np.max(X_size_mag, axis=0))

    fig = plt.figure()

    # y_pred = cluster.KMeans(n_clusters=2).fit_predict(X_size_mag)
    y_pred = cluster.SpectralClustering(
        n_clusters=2, affinity="nearest_neighbors"
    ).fit_predict(X_size_mag)

    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(X_size_mag[:, 0], X_size_mag[:, 1], c=y_pred)
    ax.set_title("Size vs mag clusters (normalized)")
    ax.set_xlabel("size")
    ax.set_ylabel("magniture")

    # X_size_mag[:, 1] = X_size_mag[:, 1]**(1 / 3)
    # X_size_mag[:, 0] = X_size_mag[:, 0]**(3)
    X_size_mag = X_size_mag * maxes
    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(X_size_mag[:, 0], X_size_mag[:, 1], c=y_pred)
    ax.set_title("Size vs mag clusters")
    ax.set_xlabel("size")
    ax.set_ylabel("magniture")

    # 3D data
    X_3 = blobs[:, [2, 3, 4]]
    X_3, maxes = normalize_features(X_3)
    y_pred = cluster.SpectralClustering(
        n_clusters=2, affinity="nearest_neighbors"
    ).fit_predict(X_3)
    # ax = fig.add_subplot(1, 3, 3, projection='3d')
    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1, projection="3d")

    X_3 = X_3 * maxes

    ax.scatter(X_3[:, 0], X_3[:, 1], X_3[:, 2], c=y_pred)
    ax.set_title("Size, mag, var clusters")
    ax.set_xlabel("size")
    ax.set_ylabel("magniture")
    ax.set_zlabel("variance")
    return y_pred
