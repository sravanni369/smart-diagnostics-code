"""
VERBATIM transcription — DO NOT EDIT, DOES NOT RUN.

Source: "Development of AI Model to Estimate the Head of an Electrical Submersible Pump"
         A. Rahman et al.
         in: Smart Diagnostics and Predictive Maintenance, ed. Aydin Azizi, Springer.
         Printed pages 134-137  (PDF pages 141-144)

The chapter states its own environment (printed p.133 / PDF p.140):
    Ubuntu 24.04.4 LTS, Python 3.11.13, TensorFlow 2.18.0, Keras 3.8.0,
    keras-tuner 1.4.7, Intel Xeon @ 2.20GHz.

Kept unmodified as the reference point for the audit in ISSUES_FOUND.md.
Note issue C6-1: the ColumnTransformer silently reorders columns, so the
`columns=input_columns` labels below are wrong. That bug is preserved here.
"""

# ----------------------------------------------------------------------------------
# Appendix 2: Data Preprocessing             (printed p.134 / PDF p.141)
# ----------------------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

df = pd.read_csv("dataset.csv")
rnd_seed = 1746448054

input_transformer = ColumnTransformer(transformers=[
    ('std', StandardScaler(), ["Speed", "Flowrate"]),
    ('robust', RobustScaler(), ["Viscosity"])
])
output_transformer = MinMaxScaler()

input_columns = ["Speed", "Viscosity", "Flowrate"]
output_columns = ["Head"]

training_df, buffer_df = train_test_split(df, test_size=0.3, random_state=rnd_seed)
validation_df, testing_df = train_test_split(buffer_df, test_size=0.5, random_state=rnd_seed)

training_input_df, training_output_df = training_df[input_columns], training_df[output_columns]
validation_input_df, validation_output_df = validation_df[input_columns], validation_df[output_columns]
testing_input_df, testing_output_df = testing_df[input_columns], testing_df[output_columns]

training_input_df_norm = pd.DataFrame(
    input_transformer.fit_transform(training_input_df), columns=input_columns)
training_output_df_norm = pd.DataFrame(
    output_transformer.fit_transform(training_output_df), columns=output_columns)
validation_input_df_norm = pd.DataFrame(
    input_transformer.transform(validation_input_df), columns=input_columns)
validation_output_df_norm = pd.DataFrame(
    output_transformer.transform(validation_output_df), columns=output_columns)
testing_input_df_norm = pd.DataFrame(
    input_transformer.transform(testing_input_df), columns=input_columns)


# ----------------------------------------------------------------------------------
# Appendix 3: Preliminary Model              (printed p.135 / PDF p.142)
# ----------------------------------------------------------------------------------

"""
The preliminary MLP model is created using the following block of Python code
"""
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.Input((3,), name="layers_input"),
    tf.keras.layers.Dense(10, activation="tanh", name="layers_hidden_1"),
    tf.keras.layers.Dense(10, activation="relu", name="layers_hidden_2"),
    tf.keras.layers.Dense(1, activation="tanh", name="layers_output")
], name="MLP")

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

"""
The preliminary FCC model is created using the following block of Python code
"""
import tensorflow as tf

input_layer = tf.keras.layers.Input(shape=(3,), name="layers_input")
hidden_layer_1 = tf.keras.layers.Dense(10, activation="relu",
                                       name="layers_hidden_1")(input_layer)
hidden_layer_2_concat = tf.keras.layers.Concatenate()([input_layer, hidden_layer_1])
hidden_layer_2 = tf.keras.layers.Dense(10, activation="relu",
                                       name="layers_hidden_2")(hidden_layer_2_concat)
output_layer_concat = tf.keras.layers.Concatenate()(
    [input_layer, hidden_layer_1, hidden_layer_2])
output_layer = tf.keras.layers.Dense(1, name="ouput_layer")(output_layer_concat)

model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name="FCC")

model.compile(optimizer="adam", loss="mse", metrics=["mae"])


# ----------------------------------------------------------------------------------
# Appendix 4: Hyperparameter Tuners — MLP    (printed p.136 / PDF p.143)
# ----------------------------------------------------------------------------------

"""
The following block of code is used to generate MLP models from the given
hyperparameter space
"""
import tensorflow as tf
from keras_tuner import HyperParameters
from keras_tuner.tuners import RandomSearch, BayesianOptimization


def build_model_mlp(hp: HyperParameters):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(3,)))

    for num_layer in range(hp.Int("num_layers", 1, 2)):
        model.add(tf.keras.layers.Dense(
            units=hp.Int(f"units_{num_layer}", min_value=3, max_value=10, step=1),
            activation=hp.Choice(f"activation_{num_layer}", ["relu", "tanh"]),
            name=f"layers_hidden_{num_layer}"
        ))

    model.add(tf.keras.layers.Dense(
        1, activation=hp.Choice(f"activation_output", ["relu", "tanh"])))

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    return model


# ----------------------------------------------------------------------------------
# Appendix 4: Hyperparameter Tuners — FCC    (printed p.137 / PDF p.144)
# ----------------------------------------------------------------------------------

"""
The following block of code is used to generate FCC models from the given
hyperparameter space
"""
import tensorflow as tf
from keras_tuner import HyperParameters
from keras_tuner.tuners import RandomSearch, BayesianOptimization


def build_model_fcc(hp: HyperParameters):
    inputs = tf.keras.layers.Input(shape=(3,))
    layers = []

    for num_layer in range(hp.Int("num_layers", 1, 2)):
        units = hp.Int(f"units_{num_layer}", min_value=3, max_value=10, step=1)
        activation = hp.Choice(f"activation_{num_layer}", ["relu", "tanh"])
        if layers:
            concat = tf.keras.layers.Concatenate()(layers + [inputs])
        else:
            concat = inputs
        dense = tf.keras.layers.Dense(units, activation=activation)(concat)
        layers.append(dense)

    final_concat = tf.keras.layers.Concatenate()(layers + [inputs])
    output = tf.keras.layers.Dense(
        1, activation=hp.Choice(f"activation_output", ["relu", "tanh"]))(final_concat)

    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    return model
