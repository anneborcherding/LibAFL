from pipeline.preprocessing.util import scale_packet_values, scale_packet_length
import numpy as np
import keras.backend as K


def preprocess_autoencoder(data, encoder, sample_size):
    """
    Data is expected to be a dict of lists of binary packets.
    Returns a list of list of preprocessed packets.
    """
    result = []

    for key, pkts in data.items():
        pkts_scaled = [scale_packet_values(scale_packet_length(pkt,sample_size)) for pkt in pkts]
        sequence = encoder.predict(np.array(pkts_scaled), verbose = 0)
        result.append(sequence)
    return result

def preprocess_chiu(data, encoder, sample_size):
    """
    Data is expected to be a dict of lists of binary packets.
    Returns a list of list of preprocessed packets.
    """
    result = []

    for key, pkts in data.items():
        pkts_scaled = [scale_packet_values(scale_packet_length(pkt,sample_size)) for pkt in pkts]
        sequence = encoder.predict(np.array(pkts_scaled), verbose = 0)
        K.clear_session()
        result.append(sequence)
    return result


def preprocess_pca(data, model, sample_size):
    """
    Data is expected to be a dict of lists of binary packets.
    Returns a list of list of preprocessed packets.
    """
    result = []
    for key, pkts in data.items():
        sequence = []
        for pkt in pkts:
            pkt_scaled = scale_packet_values(scale_packet_length(pkt,sample_size))
            sequence.append(model.transform(np.array([pkt_scaled])))
        result.append(np.array([np.concatenate(pkt) for pkt in sequence]))
    return result

def preprocess(data, model, model_type:str, sample_size):
    if model_type == "pca":
        return preprocess_pca(data, model, sample_size)
    elif model_type == "ae":
        return preprocess_autoencoder(data, model, sample_size)
    elif model_type == "chiu":
        return preprocess_chiu(data, model, sample_size)
    else:
        raise ValueError("Please enter either 'pca', 'ae', or 'chiu' as model_type.")