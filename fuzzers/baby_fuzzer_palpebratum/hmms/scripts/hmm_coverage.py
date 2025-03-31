"""
This is a script which runs the whole preprocessing and behavior approximation in one step.
It can be used to be called from LibAFL to use the
"""
from pomegranate import HiddenMarkovModel

from pipeline.load_data import load_and_split_data_syn
from pipeline.preprocessing.util import bytes_as_bits_list, scale_packet_values, scale_packet_length
from pipeline.preprocessing.autoencoder.network_packet_autoencoder import NetworkPacketAutoencoder
from pipeline.preprocessing.autoencoder.chiu_autoencoder import ChiuAutoencoder
from pipeline.preprocessing.preprocess import preprocess
from pipeline.coverage.coverage_calculation import calculate_coverage_single

from pathlib import Path
import tensorflow as tf

import joblib
import yaml

SAMPLE_SIZE = 1504

model = None
number_of_nodes = None
preprocessor = None
preprocessor_type = ""
ip_addr = "127.0.0.1"

initialized = False


def init(config_path: Path):
    """
    Initializes the necessary variables to keep them in memory while calculating the coverage.
    :param config_path: Path to the config file.
    """
    config = yaml.safe_load(open(config_path))
    print(config)

    global preprocessor
    global preprocessor_type
    global model
    global number_of_nodes
    global initialized

    preprocessor_type = config['preprocessor_type']
    assert preprocessor_type == "pca" or preprocessor_type == "ae" or preprocessor_type == "ae_chiu", "Please enter either pca, ae, or ae_chiu as preprocessor_type."
    preprocessor_path = config['preprocessor_path']
    if preprocessor_type == "pca":
        preprocessor = joblib.load(preprocessor_path)
    else:
        preprocessor = tf.keras.models.load_model(preprocessor_path, compile=False)
    model = HiddenMarkovModel().from_dict(joblib.load(config['model_path']))
    number_of_nodes = config['model_nodes']

    initialized = True


def calculate_coverage(pcap_path: Path) -> list:
    """
    Runs all the steps necessary to calculate the coverage achieved by the communication in the given pcap.
    :param pcap_path: Path to the pcap including the communication associated with the current fuzzing step.
    :return: The calculated coverage as boolean list.
    """
    if not initialized:
        raise AssertionError("Values need to be initialized first.")

    # load the pcap data and split it according to the ports
    data = load_and_split_data_syn(pcap_path)
    preprocessed_data = preprocess(data, preprocessor, preprocessor_type, SAMPLE_SIZE)
    assert len(preprocessed_data) == 1
    coverage = calculate_coverage_single(model, number_of_nodes, preprocessed_data)
    return coverage


#init(Path("../config.yml"))
#print(calculate_coverage(Path("../dump/example.pcap")))
