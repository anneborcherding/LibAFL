import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras import Model

LENGTH = 1024


class ChiuAutoencoder():
    def __init__(self, sample_size, encoding_size=8, number_of_hidden_layers=3, number_of_filters=(128, 64, 32)):
        """
        Build autoencoder model according to Chiu et al. (2020)
        :return: autoencoder model
        """
        # super(ChiuAutoencoder, self).__init__()
        
        # layers_list_encoder = [layers.Input(shape=(sample_size,1))]
        # # layers_list_encoder = [layers.Input(shape=(sample_size))]
        # for i in range(number_of_hidden_layers):
        #     layers_list_encoder.append(layers.Conv1D(number_of_filters[i], 3, activation='relu', padding='same'))
        #     layers_list_encoder.append(layers.MaxPooling1D(2, padding='same'))
        # layers_list_encoder.append(layers.Flatten())
        # layers_list_encoder.append(layers.Dense(encoding_size, name='bottleneck'))
        # self.encoder = tf.keras.Sequential(layers_list_encoder)

        # layers_list_decoder = []
        # layers_list_decoder.append(layers.Dense(188 * 32))
        # layers_list_decoder.append(layers.Reshape((188, 32)))
        # for i in range(number_of_hidden_layers):
        #     layers_list_decoder.append(layers.Conv1DTranspose(number_of_filters[number_of_hidden_layers - 1 - i], 3, activation='relu',
        #                                padding='same'))
        #     layers_list_decoder.append(layers.UpSampling1D(2))
        # layers_list_decoder.append(layers.Conv1DTranspose(1, 3, activation='sigmoid', padding='same'))
        # self.decoder = tf.keras.Sequential(layers_list_decoder)

        # # OLD
        input_layer = layers.Input(shape=(sample_size, 1))
        x = input_layer
        # Encoder
        for i in range(number_of_hidden_layers):
            x = layers.Conv1D(number_of_filters[i], 3, activation='relu', padding='same')(x)
            x = layers.MaxPooling1D(2, padding='same')(x)
        x = layers.Flatten()(x)
        encoding_layer = layers.Dense(encoding_size, name='bottleneck')(x)
        x = encoding_layer

        # Decoder
        x = layers.Dense(188 * 32)(x)
        x = layers.Reshape((188, 32))(x)
        for i in range(number_of_hidden_layers):
            x = layers.Conv1DTranspose(number_of_filters[number_of_hidden_layers - 1 - i], 3, activation='relu',
                                       padding='same')(x)
            x = layers.UpSampling1D(2)(x)
        decoded = layers.Conv1DTranspose(1, 3, activation='sigmoid', padding='same')(x)


        self.model = Model(input_layer, decoded)
        self.encoder = Model(self.model.input, encoding_layer)
        # self.model.compile(optimizer='adam', loss=losses.mse, metrics='accuracy')

        # for layer in self.model.layers:
        #     print(layer._name)


    def call(self, x):
        decoded = self.model.predict(x)
        # encoded = self.encoder(x)
        # decoded = self.decoder(encoded)
        return decoded
