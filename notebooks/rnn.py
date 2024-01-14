
#%%import os
import pickle
import pandas as pd
import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

pd.set_option('mode.chained_assignment', None)


from tensorflow.keras import Model, Sequential

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.layers.experimental.preprocessing import Normalization

from tensorflow.keras.preprocessing import timeseries_dataset_from_array
# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore',category=UserWarning)


#%%
os.chdir('raw_data')

dict_week = pickle.load(open('dict_week.pkl','rb'))
#%%

week_35 = dict_week[35].copy()
week_17 = dict_week[17].copy()

del dict_week[35]
del dict_week[17]

dict_32 = {}
for key, value in dict_week.items():
    if key >= 36:
        new_key = key - 35
    else:
        new_key = key + 16
    dict_32[new_key] = value

dict_verification = dict(list(dict_32.items())[-5:])
dict_for_x =  dict(list(dict_32.items())[:-5])


for k,v in dict_for_x.items():
    df = dict_for_x[k].copy()
    df = df.reset_index(drop=True)
    dict_for_x[k]= df

concatenate_df = pd.concat(dict_for_x.values())

n = len(concatenate_df)
# split 70:20:10 (train: validation: test)
train_df = concatenate_df[0:int(n*0.7)]
val_df = concatenate_df[int(n*0.7): int(n*0.9)]
test_df = concatenate_df[int(n*0.9):]
#%%

# initialize
scaler = MinMaxScaler()

# scale our training
scaler.fit(train_df)

# transform all validation and test
train_df.loc[:,train_df.columns] = scaler.transform(train_df[train_df.columns])
val_df.loc[:,val_df.columns] = scaler.transform(val_df[val_df.columns])
test_df.loc[:,test_df.columns] = scaler.transform(test_df[test_df.columns])

#%%
class DataWindow():
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df, test_df=test_df,
                 label_columns=None):

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_to_inputs_labels(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:,:,self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='energy', max_subplots=3):
        inputs, labels = self.sample_batch

        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col} [scaled]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
              label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
              label_col_index = plot_col_index

            if label_col_index is None:
              continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', marker='s', label='Labels', c='green', s=64)
            if model is not None:
              predictions = model(inputs)
              plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                          marker='X', edgecolors='k', label='Predictions',
                          c='red', s=64)

            if n == 0:
              plt.legend()

        plt.xlabel('Time (h)')

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32
        )

        ds = ds.map(self.split_to_inputs_labels)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def sample_batch(self):
        result = getattr(self, '_sample_batch', None)
        if result is None:
            result = next(iter(self.train))
            self._sample_batch = result
        return result




def compile_and_fit(model, window, patience=10, max_epochs=50, option ='earlystopping'):
    if option == 'earlystopping':
        method = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   mode='min', restore_best_weights= True)

    if option =='reduce_lr':
         method =  ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)

    model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(),
                  metrics=[MeanAbsoluteError()])

    history = model.fit(window.train,
                       epochs=max_epochs,
                       validation_data=window.val,
                       callbacks=[method])

    return history

val_performance = {}
performance = {}
multi_window = DataWindow(input_width=(144*4), label_width=144, shift=144, label_columns=['energy'])

lstm_model = models.Sequential()
lstm_model.add(layers.LSTM(64,
                        activation='tanh',
                        return_sequences = False,
                       # kernel_regularizer=L1L2(l1=0.001, l2=0.001),
                        ))
lstm_model.add(layers.Dense(1, activation='linear'))

history = compile_and_fit(lstm_model, multi_window)

val_performance['LSTM'] = lstm_model.evaluate(multi_window.val)
performance['LSTM'] = lstm_model.evaluate(multi_window.test, verbose=0)

# %%
