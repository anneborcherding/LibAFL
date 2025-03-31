"""
Script for preprocessing and model generation. It performs the following steps:
1. Load the user data and train an autoencoder, an autoencoder by Chiu, and a PCA on this data
2. Preprocess user data (if set in parameters)
3. Preprocess user data (if set in parameters)
4. Train HMMs on the preprocessed usage data (if that has been created)

This script takes the following parameters:
1. server to analyze (lightftp, bftpd, proftp, pureftpd)
2. train preprocessors (true, false)
3. data to preprocess (afl, usage, both, none)
4. train hmms (true, false)

The paths to be used to load and save data can be set using the constant variables below.
"""
import os
import sys

# necessary to load modules from sibling folder
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import numpy as np
import tensorflow as tf

from os import listdir

import joblib
import time

from sklearn.model_selection import train_test_split
from scapy.all import raw
from sklearn.decomposition import PCA
from pathlib import Path

from pipeline.load_data import load_data, load_and_split_data_syn
from pipeline.preprocessing.util import scale_packet_values, scale_packet_length
from pipeline.preprocessing.autoencoder.network_packet_autoencoder import NetworkPacketAutoencoder
from pipeline.preprocessing.autoencoder.chiu_autoencoder import ChiuAutoencoder
from pipeline.preprocessing.preprocess import preprocess_pca, preprocess_chiu, preprocess_autoencoder
from pipeline.coverage.hmms import create_model, save_model, save_model_image

ftp_server = sys.argv[1]
train_preprocessors = sys.argv[2] == "true"
data = sys.argv[3]
train_hmms = sys.argv[4] == "true"

preprocess_user_data = False
preprocess_afl_data = False

if data == "afl":
    preprocess_afl_data = True
elif data == "user":
    preprocess_user_data = True
elif data == "both":
    preprocess_user_data = True
    preprocess_afl_data = True

SAMPLE_SIZE = 1504
NO_NODES_HMMS = [7,18, 27, 38, 51]

IP_ADDRESS_TARGET = "127.0.0.1"
IP_ADDRESS_SENDER = "127.0.0.1"

USER_DATA = Path(f"../data/user_data/{ftp_server}.pcap")
AFL_DATA_FOLDER = Path(f"../data")
SAVE_FOLDER_AFL = Path(f"../data/preprocessed/{ftp_server}/afl")
SAVE_FOLDER_USER = Path(f"../data/preprocessed/{ftp_server}/user")
SAVE_FOLDER_PREPROCESSORS = Path(f"../models/{ftp_server}/preprocessors")
SAVE_FOLDER_MODELS = Path(f"../models/{ftp_server}/hmms")

# change path if you would like to load the preprocessed data from somewhere else
LOAD_FOLDER_USER = SAVE_FOLDER_USER

SAVE_FOLDER_AFL.mkdir(parents=True, exist_ok=True)
SAVE_FOLDER_USER.mkdir(parents=True, exist_ok=True)
SAVE_FOLDER_PREPROCESSORS.mkdir(parents=True, exist_ok=True)
SAVE_FOLDER_MODELS.mkdir(parents=True, exist_ok=True)

print(f"I will load the user data from {USER_DATA} and the AFL data from {AFL_DATA_FOLDER}.\n"
      f"I assume that the AFL data files have the string 'afl' in their name.\n"
      f"I will save the trained models to {SAVE_FOLDER_MODELS} and the preprocessed AFL data to {SAVE_FOLDER_AFL}.\n")

if not USER_DATA.is_file():
    raise FileExistsError(f"USER_DATA should point to an existing file. Currently, it points to {USER_DATA}")

####################################################
# Preprocessor training
####################################################

if train_preprocessors:
    print("[INFO] Starting training of preprocessors")

    packets_filtered = load_data(USER_DATA)
    packets_binary = [raw(pkt) for pkt in packets_filtered]
    data = np.array([scale_packet_values(scale_packet_length(pkt, SAMPLE_SIZE)) for pkt in packets_binary])
    train_data, test_data = train_test_split(data, test_size=0.2)

    print("[INFO] Training autoencoder")

    start_ae = time.time()
    autoencoder = NetworkPacketAutoencoder(sample_size=SAMPLE_SIZE, encoding_size=8)
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.mse)
    history_ae = autoencoder.fit(train_data, train_data,
                                 epochs=100,
                                 validation_data=(test_data, test_data),
                                 shuffle=True,
                                 verbose=0)
    end_ae = time.time()
    autoencoder.save(SAVE_FOLDER_PREPROCESSORS.joinpath(f"ae_{ftp_server}"))
    autoencoder.encoder.save(SAVE_FOLDER_PREPROCESSORS.joinpath(f"ae_encoder_{ftp_server}"))

    print("[INFO] Training Chiu's autoencoder")

    start_chiu = time.time()
    autoencoder_chiu = ChiuAutoencoder(sample_size=SAMPLE_SIZE, encoding_size=8)
    autoencoder_chiu.model.compile(optimizer='adam', loss=tf.keras.losses.mse)
    history_chiu = autoencoder_chiu.model.fit(train_data, train_data,
                                              epochs=100,
                                              validation_data=(test_data, test_data),
                                              shuffle=True,
                                              verbose=0)
    autoencoder_chiu.model.save(SAVE_FOLDER_PREPROCESSORS.joinpath(f"ae_chiu_{ftp_server}"))
    autoencoder_chiu.encoder.save(SAVE_FOLDER_PREPROCESSORS.joinpath(f"ae_chiu_encoder_{ftp_server}"))
    end_chiu = time.time()

    print("[INFO] Training PCA")

    start_pca = time.time()
    pca = PCA(n_components=8)
    pca.fit(data)
    end_pca = time.time()

    joblib.dump(pca, SAVE_FOLDER_PREPROCESSORS.joinpath(f"pca_{ftp_server}"))

    with open(SAVE_FOLDER_PREPROCESSORS.joinpath("training_time.csv"), 'w') as f:
        f.write(f"AE, {end_ae-start_ae}\n")
        f.write(f"CHIU, {end_chiu-start_chiu}\n")
        f.write(f"PCA, {end_pca-start_pca}\n")

    print("[INFO] Finished training and saving models")
else:
    print("[INFO] Skipped training of preprocessors")

####################################################
# Preprocess user data using the new preprocessors
####################################################

if preprocess_user_data:
    print("[INFO] Preprocessing user data")

    autoencoder = tf.keras.models.load_model(SAVE_FOLDER_PREPROCESSORS.joinpath(f"ae_encoder_{ftp_server}"), compile=False)
    autoencoder_chiu = tf.keras.models.load_model(SAVE_FOLDER_PREPROCESSORS.joinpath(f"ae_chiu_encoder_{ftp_server}"), compile=False)
    pca = joblib.load(SAVE_FOLDER_PREPROCESSORS.joinpath(f"pca_{ftp_server}"))

    user_data_filtered_split = load_and_split_data_syn(USER_DATA)

    user_data_preprocessed_autoencoder = preprocess_autoencoder(user_data_filtered_split, autoencoder, SAMPLE_SIZE)
    user_data_preprocessed_chiu = preprocess_chiu(user_data_filtered_split, autoencoder_chiu, SAMPLE_SIZE)
    user_data_preprocessed_pca = preprocess_pca(user_data_filtered_split, pca, SAMPLE_SIZE)

    joblib.dump(user_data_preprocessed_pca, SAVE_FOLDER_USER.joinpath(f"preprocessed_user_data_pca_{ftp_server}"))
    joblib.dump(user_data_preprocessed_autoencoder,
                SAVE_FOLDER_USER.joinpath(f"preprocessed_user_data_ae_{ftp_server}"))
    joblib.dump(user_data_preprocessed_chiu, SAVE_FOLDER_USER.joinpath(f"preprocessed_user_data_chiu_{ftp_server}"))
else:
    print("[INFO] Skipped preprocessing of usage data")

####################################################
# Preprocess afl data using the new preprocessors
####################################################

if preprocess_afl_data:
    print("[INFO] Preprocessing afl data")
    afl_files = [f for f in listdir(AFL_DATA_FOLDER) if "afl_" + ftp_server in f]
    autoencoder = tf.keras.models.load_model(SAVE_FOLDER_PREPROCESSORS.joinpath(f"ae_encoder_{ftp_server}"), compile=False)
    autoencoder_chiu = tf.keras.models.load_model(SAVE_FOLDER_PREPROCESSORS.joinpath(f"ae_chiu_encoder_{ftp_server}"), compile=False)
    pca = joblib.load(SAVE_FOLDER_PREPROCESSORS.joinpath(f"pca_{ftp_server}"))

    index = 0
    for file in afl_files:
        print("[INFO] Starting with round " + str(index))

        afl_data = load_and_split_data_syn(AFL_DATA_FOLDER.joinpath(file))

        print("[INFO] Preprocessing with Autoencoder")
        afl_data_ae = preprocess_autoencoder(afl_data, autoencoder, SAMPLE_SIZE)

        print("[INFO] Preprocessing with PCA")
        afl_data_pca = preprocess_pca(afl_data, pca, SAMPLE_SIZE)

        print("[INFO] Preprocessing with Chiu Autoencoder")
        afl_data_chiu = preprocess_chiu(afl_data, autoencoder_chiu, SAMPLE_SIZE)

        joblib.dump(afl_data_ae, SAVE_FOLDER_AFL.joinpath(f"preprocessed_afl_data_ae_{ftp_server}_{str(index)}"))
        joblib.dump(afl_data_pca, SAVE_FOLDER_AFL.joinpath(f"preprocessed_afl_data_pca_{ftp_server}_{str(index)}"))
        joblib.dump(afl_data_chiu, SAVE_FOLDER_AFL.joinpath(f"preprocessed_afl_data_chiu_{ftp_server}_{str(index)}"))

        index += 1
else:
    print("[INFO] Skipped preprocessing of afl data")

####################################################
# Generate Hidden Markov Models
####################################################
if train_hmms:
    preprocessor_names = ["pca", "ae", "chiu"]

    pca_data = joblib.load(LOAD_FOLDER_USER.joinpath(f"preprocessed_user_data_pca_{ftp_server}"))
    ae_data = joblib.load(LOAD_FOLDER_USER.joinpath(f"preprocessed_user_data_ae_{ftp_server}"))
    chiu_data = joblib.load(LOAD_FOLDER_USER.joinpath(f"preprocessed_user_data_chiu_{ftp_server}"))

    data = {preprocessor_names[0]: pca_data, preprocessor_names[1]: ae_data, preprocessor_names[2]: chiu_data}

    skipped_models = []
    stats = ""

    for no in NO_NODES_HMMS:
        print(f"[INFO] Generating models with {no} nodes.")
        for preprocessor in preprocessor_names:
            print(f"[INFO] Now creating models for {preprocessor}")
            start_hmm = time.time()
            model = create_model(no, data[preprocessor])
            end_hmm = time.time()
            id = f"hmm_{ftp_server}_{preprocessor}_{no}"
            if model:
                model.name = id
                save_model(model, SAVE_FOLDER_MODELS.joinpath(id))
                save_model_image(model, SAVE_FOLDER_MODELS, id)
            else:
                skipped_models.append(id)
            stats += f"[{id}, {end_hmm-start_hmm}\n"
    if len(skipped_models) > 0:
        print(f"[WARNING] Was not able to generate the following models: {skipped_models}")
    else:
        print("[IMPORTANT INFO] Ignore above np.core or np.linalg exceptions if there are any.")
    with open(SAVE_FOLDER_MODELS.joinpath("training_time.csv"), 'w') as f:
        f.write(stats)
else:
    print("[INFO] Skipped training of HMMs")

print("[INFO] Finished.")