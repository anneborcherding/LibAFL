import joblib
import os
import numpy as np

def generate_coverage_map(viterbi_res, number_of_nodes):
    """Calculates the edge coverage. Expects the estimated nodes as well as the absolute number of nodes.
    Returns a coverage list which is set to 1 if the corresponding edge has been hit as least once."""
    # maybe we would want to check if transition probabilities 0 exist: model.transmat_
    estimated_nodes = [i[0] for i in viterbi_res[1]]
    coverage = [0 for i in range(number_of_nodes * number_of_nodes)]
    for i in range(len(estimated_nodes)-1):
        # path just set to one if hit
        # maybe we could also sum up the hits
        coverage[estimated_nodes[i]*number_of_nodes + estimated_nodes[i+1]] = 1
    return coverage

def calculate_coverage(model, no_components, files, preprocessed_data_folder):
    coverage_summarized = [0 for i in range(no_components * no_components)]
    cov_over_time = []

    round_ctr = 0
    for file in files:
        round_ctr += 1
        print(f"Round {round_ctr}")
        data = joblib.load(os.path.join(preprocessed_data_folder, file))
        for seq in data:
            estimated_nodes = model.viterbi(seq)
            coverage = generate_coverage_map(estimated_nodes, no_components)
            coverage_summarized = np.logical_or(coverage_summarized, coverage)
            cov_over_time.append(sum(coverage_summarized)/(no_components * no_components))
    return coverage_summarized, cov_over_time

def calculate_coverage_single(model, no_components, preprocessed_data):
    if len(preprocessed_data) != 1: print(f"[Warning] The length of the preprocessed data was not 1 but {len(preprocessed_data)}.")
    estimated_nodes = model.viterbi(preprocessed_data[0])
    return generate_coverage_map(estimated_nodes, no_components)