import numpy as np


def get_data(addr: str) -> []:
    points = []
    with open(addr, "r") as file:
        lines = file.readlines()
        for line in lines:
            data = []
            split = line.strip().split(",")
            for i in range(len(split)):
                if split[i] == "":
                    continue
                if split[i] == "?":
                    data = []
                    break
                data.append(split[i])
            if len(data) > 0:
                points.append(data)
    return np.array(points)


def calculate_probabilities(data: []) -> []:
    which_is_where = []
    outputs = {}
    for i in range(len(data)):
        element = data[i]
        for j in range(len(element)):
            attribute_value = element[j]
            output = element[-1]
            if output not in outputs.keys():
                outputs[output] = 0
            if len(which_is_where) <= j:
                which_is_where.append({})
            if attribute_value not in which_is_where[j].keys():
                which_is_where[j][attribute_value] = {}
            if output not in which_is_where[j][attribute_value].keys():
                which_is_where[j][attribute_value][output] = 0

            outputs[output] += 1
            which_is_where[j][attribute_value][output] += 1
    which_is_where = np.array(which_is_where)
    for i in range(len(which_is_where)):
        element = which_is_where[i]
        for key in element.keys():
            for output_key in outputs.keys():
                if output_key not in which_is_where[i][key].keys():
                    which_is_where[i][key][output_key] = 1
                which_is_where[i][key][output_key] = (
                    which_is_where[i][key][output_key] + 1
                ) / (outputs[output_key] + 1)

    return [which_is_where, outputs]


def convert_to_probabilities(outputs: {}) -> dict:
    all = 0
    for key in outputs.keys():
        all += outputs[key]
    for key in outputs.keys():
        outputs[key] /= all
    return outputs


def __get_class(single: [], probabilities: [], outputs: {}) -> str:
    outputs_probabilities = {}
    valid_output = ""
    valid_probability = 0.0
    for output_key in outputs.keys():
        outputs_probabilities[output_key] = 0.0
        for i in range(len(single) - 1):
            if outputs_probabilities[output_key] == 0:
                outputs_probabilities[output_key] = probabilities[i][single[i]][
                    output_key
                ]
                continue
            outputs_probabilities[output_key] *= probabilities[i][single[i]][output_key]
        outputs_probabilities[output_key] *= outputs[output_key]
        if outputs_probabilities[output_key] > valid_probability:
            valid_probability = outputs_probabilities[output_key]
            valid_output = output_key
    return valid_output


def test(data: [], probabilities: [], outputs: {}) -> float:
    outputs = convert_to_probabilities(outputs)
    truth = 0
    for i in range(len(data)):
        single = data[i]
        if __get_class(single, probabilities, outputs) == single[-1]:
            truth += 1
    return round(truth / len(data) * 100, 2)
