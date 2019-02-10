import pandas as pd
import tensorflow as tf
import numpy as np
import math
import os

from tensorflow.python import debug as tf_debug
from sklearn import metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# TODO:

# Save model to file



###########################
### Tweakable variables ###
###########################

# Total amount of data points to use for training and validation
data_points = 500
# Proportion of total data D that is validation V (training data left is D - (V*D) )
val_data_proportion = 0.2

feature_fields = ['Gender', 'Age', 'City_Category', 'Occupation', 'Marital_Status', 'Stay_In_Current_City_Years',
        'Product_Category_1', 'Product_Category_2', 'Product_Category_3']

# Training parameters
training_steps = 50
batch_size = 20
learning_rate = 0.0001
hidden_layers = [50,30,30]

# Amount of times to segment training to output current state
training_periods = 5


####################
# Static variables #
####################

train_eval_data_path = "data/train_eval_data.csv"

def format_features(df):
    # Remove UserID and Product_ID as they do not help in seeing patterns
    # TODO: Possibly tweak the Product_ID column to remove any products appearing too few times
    data = pd.DataFrame()
    
    # Copy chosen fields from df dataframe
    for f in feature_fields:
        data[f] = df[f]

    # Replace NA entries with 0 and change type to int32
    for i in ['Product_Category_1', 'Product_Category_2', 'Product_Category_3']:
        data[i] = data[i].fillna(0).astype('int32')

    return data

def format_labels(df):
    data = pd.DataFrame()
    data['Purchase'] = df['Purchase']
    return data

def create_feature_cols(features):
    cols = []
    # Columns with strings as features
    voc_cols = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']
    id_cols = ['Marital_Status', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3']
    for c in voc_cols:
        cols.append(tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                key=c,
                vocabulary_list=features[c].unique().tolist()
        )))
    for c in id_cols:
        cols.append(tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_identity(
                key=c,
                num_buckets=features[c].max()+1
        )))
    return cols

def train_input_fn(features, labels, batch_size=1, shuffle=True):
  #  features = {key:np.array(value) for key,value in dict(features).items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    ds = ds.batch(batch_size)
    if shuffle:
        # TODO: Try removing repeat
        ds = ds.shuffle(10000)
    return ds

# TODO: Gor om features till feature_columns
def bf_model_fn(features, labels, mode, params):
    current_layer = tf.feature_column.input_layer(features, params["feature_columns"])
    for layer_dim in params["hidden_units"]:
        current_layer = tf.layers.dense(current_layer, layer_dim)

    output_layer = tf.layers.dense(current_layer, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'Price': output_layer}
        export_outputs = {'prediction': tf.estimator.export.PredictOutput(predictions)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions,export_outputs=export_outputs)

    loss = tf.math.sqrt(tf.losses.mean_squared_error(labels, output_layer))
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, 
            loss=loss
        )

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

# Read data
df = pd.read_csv(train_eval_data_path)

if data_points > len(df):
    print("NOTE: data_points value larger than total data. Setting to %i" % len(df))
    data_points = len(df)

# Ensure that validation data not set to higher than 1 (not that it would make much sense to have it set to 1)
val_data_proportion = min(val_data_proportion,1)

# Format training data
training_data = df.head(int((1-val_data_proportion) * data_points))
training_feats = format_features(training_data)
training_labels = format_labels(training_data)

# Format validation data
validation_data = df.tail(int(val_data_proportion * data_points))
validation_feats = format_features(validation_data)
validation_labels = format_labels(validation_data)

# Define estimator
bf_model = tf.estimator.Estimator(
    model_fn=bf_model_fn,
    params={
        "feature_columns": create_feature_cols(df),
        "hidden_units": hidden_layers,
        "learning_rate": learning_rate
    })

# Training loop
for i in range(training_periods):
    steps = training_steps / training_periods
    print("Training with steps %i ..." % steps)
    bf_model.train(
        input_fn=lambda:train_input_fn(training_feats, training_labels, batch_size=batch_size),
        steps=steps
    )
    print("Evaluating ...")
    eval_res = bf_model.evaluate(
        input_fn=lambda:train_input_fn(validation_feats, validation_labels, batch_size=batch_size)
    )
    print("Current evaluation loss: %i" % eval_res['loss'])

print("Final evaluation loss: %i" % eval_res['loss'])



def serving_input_receiver_fn(default_batch_size=None):
    feature_spec = {
        'Gender': tf.FixedLenFeature([], tf.string),
        'Age': tf.FixedLenFeature([], tf.string),
        'City_Category': tf.FixedLenFeature([], tf.string),
        'Occupation': tf.FixedLenFeature([], tf.int64),
        'Marital_Status': tf.FixedLenFeature([], tf.int64),
        'Stay_In_Current_City_Years': tf.FixedLenFeature([], tf.string),
        'Product_Category_1': tf.FixedLenFeature([], tf.int64),
        'Product_Category_2': tf.FixedLenFeature([], tf.int64),
        'Product_Category_3': tf.FixedLenFeature([], tf.int64),
    }
    example = tf.placeholder(
      shape=[default_batch_size],
      dtype=tf.string,
    )
    features = tf.parse_example(example, feature_spec)
    receiver_tensors = {'example': example}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

bf_model.export_saved_model(".", serving_input_receiver_fn)