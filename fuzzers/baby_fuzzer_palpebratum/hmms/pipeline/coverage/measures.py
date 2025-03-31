import numpy as np
import similaritymeasures as sm
import math
from sklearn.metrics import mean_squared_error

def normalize(curve):
    curve = [0.0] + curve
    min_val = min(curve)
    max_val = max(curve)
    if min_val == max_val:
        return [0.0] + [1.0 for i in range(len(curve) - 1)]
    return [(x - min_val) / (max_val - min_val) for x in curve[1:]]


def calculate_measures(sequence1, sequence2):
    """
    Calculate lots of different curve similarity measures for the two given sequences.
    """
    xs = range(0, len(sequence1))
    data1 = np.zeros((len(sequence1), 2))
    data1[:, 0] = xs
    data1[:, 1] = sequence1

    data2 = np.zeros((len(sequence2), 2))
    data2[:, 0] = xs
    data2[:, 1] = sequence2

    # print("Calculate partial curve mapping")
    pcm = sm.pcm(data1, data2)
    # print("Calculate frechet disctance")
    df = sm.frechet_dist(data1, data2)
    # print("Calculate area between curves")
    area = sm.area_between_two_curves(data1, data2)
    # print("Calculate curve length")
    cl = sm.curve_length_measure(data1, data2)
    # print("Calculate dynmic time warping")
    dtw = sm.dtw(data1, data2)[0]
    # print("Calculate mean squared error")
    mse = math.sqrt(mean_squared_error(sequence1, sequence2))
    # print("Calculate correlation")
    correlation = np.corrcoef(sequence1, sequence2)[0][1]

    return pcm, df, area, cl, dtw, mse, correlation


def calculate_coverage_similarity(fuzzer, hmm, max_distance=5):
    """
    Our own coverage similarity calculation.
    """
    fuzzer = normalize(fuzzer)
    hmm = normalize(hmm)

    if not len(fuzzer) == len(hmm):
        print(len(fuzzer))
        print(len(hmm))
    test_case_number = min(len(fuzzer), len(hmm))
    gradient_fuzzer = np.gradient(fuzzer)
    gradient_hmm = np.gradient(hmm)

    gradient_number = min(len(gradient_fuzzer), len(gradient_hmm))
    # 1: Get bitmap of increases
    bitmap_fuzzer = [1 if gradient_fuzzer[i] > 0 else 0 for i in range(gradient_number)]
    bitmap_hmm = [1 if gradient_hmm[i] > 0 else 0 for i in range(gradient_number)]

    number_increases_fuzzer = len(list(filter(lambda x: x > 0, bitmap_fuzzer)))
    number_increases_hmm = len(list(filter(lambda x: x > 0, bitmap_hmm)))

    print("Difference in coverage increases ", abs(number_increases_fuzzer - number_increases_hmm))

    # 2: Match increases to each other
    indices_fuzzer = [i for i in range(gradient_number) if bitmap_fuzzer[i] > 0]
    indices_hmm = [i for i in range(gradient_number) if bitmap_hmm[i] > 0]
    number_indices = len(indices_fuzzer) + len(indices_hmm)
    print("Number of fuzzing increases", len(indices_fuzzer))
    print("Number of hmm increases", len(indices_hmm))
    index_fuzzer = 0
    index_hmm = 0
    matched = []
    unmatched_fuzzer = []
    unmatched_hmm = []

    possible_matches = []
    for index_fuzzer in range(len(indices_fuzzer)):
        current_fuzzer = indices_fuzzer[index_fuzzer]
        for index_hmm in range(len(indices_hmm)):
            current_hmm = indices_hmm[index_hmm]
            if current_fuzzer > (current_hmm + max_distance): continue
            if current_fuzzer + max_distance < current_hmm: break
            possible_matches.append((current_fuzzer, current_hmm, abs(current_fuzzer - current_hmm)))

    # greedy matching
    sorted_matches = sorted(possible_matches, key=lambda x: x[2])
    unmatched_fuzzer = indices_fuzzer
    unmatched_hmm = indices_hmm
    matches = []
    for index_fuzzer, index_hmm, dist in sorted_matches:
        if index_fuzzer in unmatched_fuzzer and index_hmm in unmatched_hmm:
            matches.append((index_fuzzer, index_hmm, dist))
            unmatched_fuzzer.remove(index_fuzzer)
            unmatched_hmm.remove(index_hmm)

    print("Unmatched fuzzer", len(unmatched_fuzzer))
    print("Unmatched hmm", len(unmatched_hmm))
    print("Matches", len(matches))

    # Scoring
    score = 0.0
    # Matched score: Euclidean
    # normalize x direction
    for index_fuzzer, index_hmm, dist in matches:
        score += ((index_fuzzer - index_hmm) / max_distance) ** 2 + (
                    gradient_fuzzer[index_fuzzer] - gradient_hmm[index_hmm]) ** 2
    # Unmatched score: Maximum euclidean distance
    # In x: normalized max_distance -> 1
    # In y: 1
    for unmatched in unmatched_fuzzer:
        score += 2
    for unmatched in unmatched_hmm:
        score += 2

    # normalize score
    # max number of none matches is n - 2 --> assume n
    # 2 * n is the maximum
    score = score / (2 * number_indices)

    print("Score", score)

    return score


def dtw_normalized_increases(fuzzer, hmm):
    number_of_increases_fuzzer = 0
    number_of_increases_hmm = 0
    normalized_fuzzer = [0]
    normalized_hmm = [0]
    for i in range(1, len(fuzzer)):
        if fuzzer[i] > fuzzer[i - 1]:
            number_of_increases_fuzzer += 1
            normalized_fuzzer.append(number_of_increases_fuzzer)
        else:
            normalized_fuzzer.append(number_of_increases_fuzzer)
        if hmm[i] > hmm[i - 1]:
            number_of_increases_hmm += 1
            normalized_hmm.append(number_of_increases_hmm)
        else:
            normalized_hmm.append(number_of_increases_hmm)

    # calculating dtw
    xs = range(0, len(normalized_fuzzer))
    data1 = np.zeros((len(normalized_fuzzer), 2))
    data1[:, 0] = xs
    data1[:, 1] = normalized_fuzzer

    data2 = np.zeros((len(normalized_hmm), 2))
    data2[:, 0] = xs
    data2[:, 1] = normalized_hmm

    dtw = sm.dtw(data1, data2, metric="euclidean")[0]
    print("DTW on y axis", dtw)


def dtw_y(fuzzer, hmm):
    # transposing
    fuzzer = [(i, fuzzer[i]) for i in range(len(fuzzer))]
    hmm = [(i, hmm[i]) for i in range(len(hmm))]

    fuzzer = sorted(fuzzer, key=lambda x: x[1])
    hmm = sorted(hmm, key=lambda x: x[1])

    fuzzer_x = list(map(lambda x: x[1], fuzzer))
    fuzzer_y = list(map(lambda x: x[0], fuzzer))
    hmm_x = list(map(lambda x: x[1], hmm))
    hmm_y = list(map(lambda x: x[0], hmm))

    # calculating dtw
    data1 = np.zeros((len(fuzzer_x), 2))
    data1[:, 0] = fuzzer_x
    data1[:, 1] = fuzzer_y

    data2 = np.zeros((len(hmm_x), 2))
    data2[:, 0] = hmm_x
    data2[:, 1] = hmm_y

    dtw = sm.dtw(data1, data2, metric="euclidean")
    print("DTW on y axis", dtw)
