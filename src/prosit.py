from src.helper import *
from src.CONSTANTS import *
import matplotlib.pyplot as plt
import wget
import pandas as pd
import numpy as np
import h5py as h5
from random import shuffle
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, GRU, Embedding, Multiply, Attention, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as k


# Download the dataset
url = 'https://figshare.com/ndownloader/files/12506534'
#wget.download(url)

# Read the downloaded data to a dataframe
with h5.File('D:/MS2/src/holdout_hcd.hdf5', 'r') as f:
  KEY_ARRAY = ["sequence_integer", "precursor_charge_onehot", "intensities_raw"]
  KEY_SCALAR = ["collision_energy_aligned_normed", "collision_energy"]
  df = pd.DataFrame({key: list(f[key][...]) for key in KEY_ARRAY})
  for key in KEY_SCALAR:
    df[key] = f[key][...]

# Add convenience columns
df['precursor_charge'] = df.precursor_charge_onehot.map(lambda a: a.argmax() + 1)
df['sequence_maxquant'] = df.sequence_integer.map(lambda s: "".join(PROSIT_INDEXED_ALPHABET[i] for i in s if i != 0))
df['sequence_length'] = df.sequence_integer.map(lambda s: np.count_nonzero(s))

# Split the data into training, validation, and test

def split_dataframe(df,
                    unique_column,
                    ratio_training=0.8,
                    ratio_validation=0.1,
                    ratio_test=0.1):
  """
  This function splits the dataframe in three splits and makes sure that values
  of `unique_column` are unique to each of the splits. This is helpful if, for
  example, you have non-unique sequence in `unique_column` but want to ensure
  that a sequence value is unique to one of the splits.
  """

  assert ratio_training + ratio_validation + ratio_test == 1

  unique = list(set(df[unique_column]))
  n_unique = len(unique)
  shuffle(unique)

  train_split = int(n_unique * ratio_training)
  val_split = int(n_unique * (ratio_training + ratio_validation))

  unique_train = unique[:train_split]
  unique_validation = unique[train_split:val_split]
  unique_test = unique[val_split:]

  assert len(unique_train) + len(unique_validation) + len(unique_test) == n_unique

  df_train = df[df[unique_column].isin(unique_train)]
  df_validation = df[df[unique_column].isin(unique_validation)]
  df_test = df[df[unique_column].isin(unique_test)]

  assert len(df_train) + len(df_validation) + len(df_test) == len(df)

  return df_train, df_validation, df_test

df_train, df_validation, df_test = split_dataframe(df, unique_column='sequence_maxquant')

# Prepare the training data
INPUT_COLUMNS = ('sequence_integer', 'precursor_charge_onehot', 'collision_energy_aligned_normed')
OUTPUT_COLUMN = 'intensities_raw'

x_train = [np.vstack(df_train[column]) for column in INPUT_COLUMNS]
y_train = np.vstack(df_train[OUTPUT_COLUMN])

x_validation = [np.vstack(df_validation[column]) for column in INPUT_COLUMNS]
y_validation = np.vstack(df_validation[OUTPUT_COLUMN])

x_test = [np.vstack(df_test[column]) for column in INPUT_COLUMNS]
y_test = np.vstack(df_test[OUTPUT_COLUMN])

# Setup model and training parameters
DIM_LATENT = 124
DIM_EMBEDDING_IN = max(PROSIT_ALHABET.values()) + 1  # max value + zero for padding
DIM_EMBEDDING_OUT = 32
EPOCHS = 20
BATCH_SIZE = 256



# Build the model with input layers for sequence, precursor charge, and collision energy
in_sequence = Input(shape=[x_train[0].shape[1]], name="in_sequence")
in_precursor_charge = Input(shape=[x_train[1].shape[1]], name="in_precursor_charge")
in_collision_energy = Input(shape=[x_train[2].shape[1]], name="in_collision_energy")

x_s = Embedding(input_dim=DIM_EMBEDDING_IN, output_dim=DIM_EMBEDDING_OUT)(in_sequence)
x_s = GRU(DIM_LATENT)(x_s)


x_z = Dense(DIM_LATENT)(in_precursor_charge)
x_e = Dense(DIM_LATENT)(in_collision_energy)

x = Multiply()([x_s, x_z, x_e])
out_intensities = Dense(y_train.shape[1])(x)

model = Model([in_sequence, in_precursor_charge, in_collision_energy], out_intensities)



def masked_spectral_distance(true, pred):
  """ This is the loss function"""
  epsilon = k.epsilon()
  pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
  true_masked = ((true + 1) * true) / (true + 1 + epsilon)
  pred_norm = k.l2_normalize(true_masked, axis=-1)
  true_norm = k.l2_normalize(pred_masked, axis=-1)
  product = k.sum(pred_norm * true_norm, axis=1)
  arccos = tf.acos(product)
  return 2 * arccos / np.pi

def main():
  if True:
    model.compile(optimizer='Adam', loss=masked_spectral_distance)
    history = model.fit(x=x_train, y=y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                   validation_data=(x_validation, y_validation))
    model.save('model_20Epoch_att.keras')

    plt.plot(range(EPOCHS), history.history['loss'], '-', color='r', label='Training loss')
    plt.plot(range(EPOCHS), history.history['val_loss'], '--', color='r', label='Validation loss')
    plt.title(f'Training and validation loss across epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    test_spectral_angle = model.evaluate(x_test, y_test)
    test_spectral_angle


if __name__ == "__main__":
    main()



