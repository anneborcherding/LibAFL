import traceback
from pathlib import Path
from graphviz import Digraph
import joblib
import numpy as np
from pomegranate import HiddenMarkovModel
from pomegranate.distributions import MultivariateGaussianDistribution


def create_model(n_component, data, threshold=20):
    # In some cases, the model cannot be created caused by an issue of the library (https://github.com/jmschrei/pomegranate/issues/633)
    # This is why we try to create the model until it was successful or the threshold is reached
    # since it adds two nodes (start and end)
    no_hmm = n_component - 2
    model = HiddenMarkovModel()
    successful = False
    i = 0

    # print(data)

    while not successful and i < threshold:
        try:
            model = model.from_samples(MultivariateGaussianDistribution, n_components=no_hmm, X=data)
        except (np.linalg.LinAlgError, np.core._exceptions.UFuncTypeError) as e:
            # print(f"Tried to create model but failed in round {i}.")
            pass
        except Exception as e:
            print(f"Tried to create model but failed in round {i}.", traceback.print_exc())
        else:
            successful = True
        i = i + 1
    if successful:
        model.bake()
        print(f"Created model with {n_component} nodes successfully in round {i - 1}.")
        return model
    return None


def save_model(model: HiddenMarkovModel, path: Path):
    joblib.dump(model.to_dict(), path)


def load_model(path: Path):
    return HiddenMarkovModel().from_dict(joblib.load(path))


def save_model_image(model : HiddenMarkovModel, save_dir : Path, name : str):
    path = save_dir.joinpath(name+".dot")
    f = Digraph(name)
    f.attr(rankdir='LR', size='8,5')

    for state_index in range(model.state_count()):
        if "-start" in model.states[state_index].name:
            f.node(str(state_index), shape="rarrow")
        elif "-end" in model.states[state_index].name:
            continue
        #     f.node(str(state_index), shape="larrow")
        else:
            f.node(str(state_index))

    for i in range(model.state_count()):
        for j in range(model.state_count()):
            transmat = model.dense_transition_matrix()
            if round(transmat[i][j], 2) > 0.00:
                f.edge(str(i), str(j), "{:.2f}".format(transmat[i][j]))

    f.render(path, format='svg', view=False)