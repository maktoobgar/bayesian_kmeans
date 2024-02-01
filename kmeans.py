from math import inf
import numpy as np


class KMeans:
    def __init__(self, n_clusters, max_iterations=100):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.centroids = []
        self.clusters = []
        self.raw_data = []
        self.center_of_classes = []

    def calculate_distances(self, data, centroids):
        distances = []
        for _ in range(len(centroids)):
            distances.append([])
        for i in range(len(data)):
            for j in range(len(centroids)):
                distances[j].append(np.linalg.norm(data[i] - centroids[j]))
        distances = np.array(distances)
        least_distances = []
        for i in range(len(data)):
            least = inf
            which = 0
            for j in range(len(distances)):
                if least > distances[j][i]:
                    which = j
                    least = distances[j][i]
            least_distances.append(which)

        return np.array(least_distances)

    def initialize_centroids(self, data):
        np.random.shuffle(data)
        return data[: self.n_clusters]

    def test(self, data) -> float:
        true_classes = data[:, -1]
        data = data[:, :-1]
        distances = []
        for _ in range(len(self.centroids)):
            distances.append([])
        for i in range(len(data)):
            for j in range(len(self.centroids)):
                distances[j].append(np.linalg.norm(data[i] - self.centroids[j]))
        distances = np.array(distances)
        least_distances = []
        for i in range(len(data)):
            least = inf
            which = 0
            for j in range(len(distances)):
                if least > distances[j][i]:
                    which = j
                    least = distances[j][i]
            least_distances.append(which)

        return (
            np.count_nonzero(
                self.center_of_classes[np.array(least_distances)] == true_classes
            )
            / len(data)
            * 100
        )

    def update_centroids(self, data, labels):
        centroids = []
        for i in range(self.n_clusters):
            indexes = np.where(np.isin(labels, i))[0]
            group_data = np.array(data[indexes])
            sum = []
            for i in range(len(group_data)):
                if len(sum) == 0:
                    sum = group_data[i]
                    continue
                sum += group_data[i]
            sum = np.array(sum) / len(group_data)
            if len(sum) == 0:
                sum = data[0] * 0
            centroids.append(sum)
        return np.array(centroids)

    def calculate_classes(self, labels):
        center_classes = []
        for i in range(self.n_clusters):
            indexes = np.where(np.isin(labels, i))[0]
            group_data = np.array(self.raw_data[indexes])[:, -1]
            outputs, count = np.unique(group_data, return_counts=True)
            if len(count) == 0:
                count = np.array([1])
                outputs = np.array([0])
            center_classes.append(outputs[np.argmax(count)])
        self.center_of_classes = np.array(center_classes)

    def fit(self, raw_data):
        self.raw_data = raw_data
        data = raw_data[:, :-1]
        self.centroids = self.initialize_centroids(data)

        labels = []
        for _ in range(self.max_iterations):
            labels = self.calculate_distances(data, self.centroids)
            new_centroids = self.update_centroids(data, labels)

            if np.array_equal(new_centroids, self.centroids):
                break

            self.centroids = new_centroids

        self.calculate_classes(labels)


# Retrieves the attribute definitions
def get_attributes_definitions(addr: str) -> dict:
    attributes = {}
    with open(addr, "r") as file:
        lines = file.readlines()
        for line_index in range(len(lines)):
            line = lines[line_index]
            i = -1 if line_index == len(lines) - 1 else 0
            data = line.strip().split(" ")
            name = data[0]
            elements = data[1].split(",")
            attributes[name] = {}
            for element in elements:
                attributes[name][element] = i
                if i == -1:
                    i = 1
                    continue
                i += 1
    return attributes


# Retrieves and defines data from attributes_definitions
def get_data_convert(attributes_definitions: dict, addr: str) -> []:
    points = []
    with open(addr, "r") as file:
        lines = file.readlines()
        for line in lines:
            data = []
            split = line.strip().split(",")
            i = 0
            for key in attributes_definitions:
                if split[i] == "?":
                    data = []
                    break
                data.append(attributes_definitions[key][split[i]])
                i += 1
            if len(data) > 0:
                points.append(data)
    return np.array(points)
