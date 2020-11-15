from keras import Model
from keras.layers import *
from keras.regularizers import l1
from keras.utils import plot_model

import matplotlib.pyplot as plt
import numpy as np

def create_autoencoder_model(training_data):
    input_size = 79
    hidden_size0 = 70
    hidden_size1 = 60
    hidden_size2 = 50
    hidden_size3 = 40
    hidden_size4 = 30
    hidden_size5 = 20
    hidden_size6 = 10
    hidden_size7 = 5
    code_size = 2 # size of the desired compression 2D or 3D to be able to visualize

    input_img = Input(shape=(input_size,))
    input_hidden_0 = Dense(hidden_size0, activation='relu')(input_img)
    input_hidden_1 = Dense(hidden_size1, activation='relu')(input_hidden_0)
    # input_hidden_1 = Dropout(.2)(input_hidden_1)
    input_hidden_2 = Dense(hidden_size2, activation='relu')(input_hidden_1)
    input_hidden_3 = Dense(hidden_size3, activation='relu')(input_hidden_2)
    # input_hidden_3 = Dropout(.2)(input_hidden_3)
    input_hidden_4 = Dense(hidden_size4, activation='relu')(input_hidden_3)
    input_hidden_5 = Dense(hidden_size5, activation='relu')(input_hidden_4)
    # input_hidden_5 = Dropout(.2)(input_hidden_5)
    input_hidden_6 = Dense(hidden_size6, activation='relu')(input_hidden_5)
    input_hidden_7 = Dense(hidden_size7, activation='relu')(input_hidden_6)
    #activity regularizer for sparsity constraints
    code_result = Dense(code_size, activation='tanh')(input_hidden_7)

    code_input = Input(shape=(code_size,))
    output_hidden_7 = Dense(hidden_size7, activation='relu')(code_input)
    output_hidden_6 = Dense(hidden_size6, activation='relu')(output_hidden_7)
    output_hidden_5 = Dense(hidden_size5, activation='relu')(output_hidden_6)
    output_hidden_4 = Dense(hidden_size4, activation='relu')(output_hidden_5)
    output_hidden_3 = Dense(hidden_size4, activation='relu')(output_hidden_4)
    output_hidden_2 = Dense(hidden_size2, activation='relu')(output_hidden_3)
    output_hidden_1 = Dense(hidden_size1, activation='relu')(output_hidden_2)
    output_hidden_0 = Dense(hidden_size0, activation='relu')(output_hidden_1)
    output_img = Dense(input_size, activation='tanh')(output_hidden_0)

    # autoencoder = Model(input_img, output_img)
    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    # autoencoder.fit(training_data, training_data, epochs=20)

    encoder = Model(input_img, code_result)
    decoder = Model(code_input, output_img)

    input = Input(shape=(input_size,))
    code = encoder(input)
    decoded = decoder(code)

    autoencoder = Model(input, decoded)
    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(training_data, training_data, epochs=500)

    # plot_model(encoder, "./figures/autoencoder/model_encoder.png", show_shapes=True)
    # plot_model(decoder, "./figures/autoencoder/model_decoder.png", show_shapes=True)
    # plot_model(autoencoder, "./figures/autoencoder/model_autoencoder.png", show_shapes=True)

    return encoder, decoder


def verify_output(training_data, encoder, decoder, i=0):
    encoded_spike = encoder.predict(training_data[i].reshape(1, -1))
    encoded_spike = np.array(encoded_spike)
    print(encoded_spike)
    print(encoded_spike.shape)
    decoded_spike = decoder.predict(encoded_spike)
    decoded_spike = np.array(decoded_spike)
    print(decoded_spike.shape)
    plt.plot(np.arange(len(training_data[i])), training_data[i])
    plt.plot(encoded_spike, c="red", marker="o")
    plt.plot(np.arange(len(decoded_spike[0])), decoded_spike[0])
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.title(f"Verify spike {i}")
    plt.savefig('./figures/autoencoder/spike0')
    plt.show()


def create_code_numpy(spike, encoder):
    return encoder.predict(spike.reshape(1, -1))[0]


def get_codes(training_data, encoder):
    return np.apply_along_axis(create_code_numpy, 1, training_data, encoder)



