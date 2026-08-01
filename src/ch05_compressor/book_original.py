"""
VERBATIM transcription — DO NOT EDIT, DOES NOT RUN.

Source: "Data-Driven Condition Monitoring of Reciprocating Compressors Using
         Multi-Label Classification"
         A.-H. Al-Obaidani et al.
         in: Smart Diagnostics and Predictive Maintenance, ed. Aydin Azizi, Springer.
         Printed pages 98-100  (PDF pages 105-107)

This file reproduces the three appendices exactly as printed, including the original
typos ("Improting Dataset", "Sepearting", "amd Testing"). It is kept unmodified as the
reference point for the audit in ISSUES_FOUND.md.

Running this file raises NameError. That is the point: see issues C5-1, C5-2, C5-3.
The corrected, executable versions live in preprocessing.py / model.py / tuner.py.
"""

# ----------------------------------------------------------------------------------
# Appendix 1: Data Pre-processing            (printed p.98 / PDF p.105)
# ----------------------------------------------------------------------------------

# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay

# Improting Dataset
file_path = '/content/drive/MyDrive/Thesis/data (1).csv'

# Separating Numerical Data
numeric_columns = df.select_dtypes(include=['number']).columns
df_numeric = df[numeric_columns]

# Sepearting Non-numerical Data
non_numeric_columns = df.select_dtypes(exclude=['number']).columns
df_non_numeric = df[non_numeric_columns]

# Converting Classification Labels into Binary Values (0, 1)
df_non_numeric['bearings'] = df['bearings'].map({'Ok': 0, 'Noisy': 1})
df_non_numeric['wpump'] = df['wpump'].map({'Ok': 0, 'Noisy': 1})
df_non_numeric['radiator'] = df['radiator'].map({'Clean': 0, 'Dirty': 1})
df_non_numeric['exvalve'] = df['exvalve'].map({'Clean': 0, 'Dirty': 1})
df_non_numeric

# Splitting Features (Inputs) and Labels (Output)
input_data = df_numeric
output_data = df_non_numeric

# Splitting Data into Training (70%) and Temporary (30%)
input_train, input_temp, output_train, output_temp = train_test_split(
    input_data, output_data, test_size=0.3, random_state=1697639)

# Spliting Temporary Data (30%) into Validation (15%) amd Testing (15%)
input_val, input_test, output_val, output_test = train_test_split(
    input_temp, output_temp, test_size=0.5, random_state=1697639)


# ----------------------------------------------------------------------------------
# Appendix 2: Model Architecture             (printed p.99 / PDF p.106)
# ----------------------------------------------------------------------------------

# Architecture of the Neural Network Model
model = keras.Sequential([
    keras.layers.Input(shape=(20,)),
    keras.layers.Dense(41, activation='relu'),
    keras.layers.Dense(21, activation='relu'),
    keras.layers.Dense(4, activation='sigmoid')
])

# Compiling the Model
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', Precision(name='precision'),
                       Recall(name='recall'), AUC(name='auc')])

# Saving training history
history = model.fit(training_ds, epochs=100, validation_data=validation_ds)


# ----------------------------------------------------------------------------------
# Appendix 3: Hyperparameter Tuner           (printed p.100 / PDF p.107)
# ----------------------------------------------------------------------------------

def build_model(hp):
    model = keras.Sequential([
        keras.layers.Input(shape=(20,)),
        keras.layers.Dense(units=hp.Int('units_layer1', min_value=16, max_value=32, step=2),
                           activation='relu'),
        keras.layers.Dense(units=hp.Int('units_layer2', min_value=8, max_value=16, step=2),
                           activation='relu'),
        keras.layers.Dense(4, activation='sigmoid')
    ])
    # Compiling the Model
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=[Precision(name='precision'), Recall(name='recall'),
                           AUC(name='auc')])
    return model


tuner = kt.BayesianOptimization(
    build_model,
    objective='val_recall',
    max_trials=150,
    executions_per_trial=2,
    directory='MODELS40',
    project_name='bayesian_tuning'
)

# Run the search
tuner.search(training_ds, epochs=100, validation_data=validation_ds, verbose=0)
