import sys
from kmeans import KMeans, get_attributes_definitions, get_data_convert
from naive_bayse import calculate_probabilities, get_data, test


def main():
    # Prepare all
    data = get_data("files/train.txt")
    test_data = get_data("files/test.txt")
    probabilities = calculate_probabilities(data)

    print(
        f"Trained on {len(data)} Data.\nTested on {len(test_data)} Data.\nAccuracy: {test(test_data, probabilities[0], probabilities[1])}%"
    )


def kmeans():
    attributes_definitions = get_attributes_definitions("files/attributes.txt")
    data = get_data_convert(attributes_definitions, "files/train.txt")
    test = get_data_convert(attributes_definitions, "files/test.txt")

    kmeans = KMeans(15)
    kmeans.fit(data)
    print(kmeans.test(test))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "kmeans":
        kmeans()
    else:
        main()
