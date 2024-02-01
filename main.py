from naive_bayse import calculate_probabilities, get_data, test


def main():
    # Prepare all
    data = get_data("files/train.txt")
    test_data = get_data("files/test.txt")
    probabilities = calculate_probabilities(data)

    print(
        f"Trained on {len(data)} Data.\nTested on {len(test_data)} Data.\nAccuracy: {test(test_data, probabilities[0], probabilities[1])}%"
    )


if __name__ == "__main__":
    main()
