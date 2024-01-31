import keras.optimizers
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model
from keras.utils import plot_model
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from utility import norm, format_output, plot_diff, plot_metrics

df = pd.read_excel('./dataset/ENB2012_data.xlsx')
df = df.sample(frac=1).reset_index(drop=True)

train: DataFrame = None
test: DataFrame = None

train, test = train_test_split(df, test_size=0.2)
train_stats = train.describe()

train_stats.pop('Y1')
train_stats.pop('Y2')
train_stats = train_stats.transpose()
train_Y = format_output(train)
test_Y = format_output(test)

# Normalize the training and test data
norm_train_X = norm(train_stats, train)
norm_test_X = norm(train_stats, test)

input_layer = Input(shape=(len(train.columns),))
first_dense = Dense(units=128, activation=tf.nn.relu)(input_layer)
second_dense = Dense(units=128, activation=tf.nn.relu)(first_dense)

# Y1 output
y1_output = Dense(1, name='y1_output')(second_dense)

third_dense = Dense(64, activation=tf.nn.relu)(second_dense)
y2_output = Dense(1, name='y2_output')(third_dense)

model = Model(inputs=input_layer, outputs=[y1_output, y2_output])
print(model.summary())
plot_model(model, show_shapes=True, show_layer_names=True, to_file='c1-1-multi-output-model.png')
model.compile(optimizer=keras.optimizers.SGD(0.001),
              loss={'y1_output': keras.losses.mse, 'y2_output': keras.losses.mse},
              metrics={'y1_output': 'mse', 'y2_output': 'mse'}
              )

history = model.fit(norm_train_X, train_Y, epochs=500, batch_size=10, validation_data=(norm_test_X, test_Y))

loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse = model.evaluate(x=norm_test_X, y=test_Y)
print(
    "Loss = {}, Y1_loss = {}, Y1_mse = {}, Y2_loss = {}, Y2_mse = {}".format(loss, Y1_loss, Y1_rmse, Y2_loss, Y2_rmse))

Y_pred = model.predict(norm_test_X)
plot_diff(test_Y[0], Y_pred[0], title='Y1')
plot_diff(test_Y[1], Y_pred[1], title='Y2')
plot_metrics(history, metric_name='y1_output_mse', title='Y1 RMSE', ylim=6)
plot_metrics(history, metric_name='y2_output_mse', title='Y2 RMSE', ylim=7)
