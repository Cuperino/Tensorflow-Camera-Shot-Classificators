#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import shot_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    # Debug
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = shot_data.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[45, 45],
        # The model must choose between 7 classes.
        n_classes=7)

    # Train the Model.
    classifier.train(
        input_fn=lambda:shot_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:shot_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['ELS', 'LS', 'MLS', 'MS', 'MCU', 'CU', 'ECU']
    predict_x = {
        'frame_height': [224.0/100, 256.0/100, 256.0/100, 256.0/100, 224.0/100, 256.0/100, 224.0/100, 256.0/100, 256.0/100, 256.0/100, 224.0/100],
        'face_x_min':   [254.0/100, 154.0/100, 108.0/100, 142.0/100, 197.0/100, 192.0/100, 252.0/100, 168.0/100, 1.0/100,   182.0/100, 246.0/100],
        'face_y_min':   [1.0/100,   25.0/100,  88.0/100,  51.0/100,  25.0/100,  21.0/100,  40.0/100,  35.0/100,  1.0/100,   19.0/100,  62.0/100 ],
        'face_x_max':   [427.0/100, 167.0/100, 155.0/100, 204.0/100, 244.0/100, 216.0/100, 303.0/100, 270.0/100, 434.0/100, 235.0/100, 258.0/100],
        'face_y_max':   [192.0/100, 47.0/100,  125.0/100, 116.0/100, 90.0/100,  59.0/100,  114.0/100, 218.0/100, 256.0/100, 113.0/100, 75.0/100 ],
        'person_x_min': [186.0/100, 144.0/100, 61.0/100,  58.0/100,  135.0/100, 165.0/100, 196.0/100, 145.0/100, 1.0/100,   108.0/100, 232.0/100],
        'person_y_min': [1.0/100,   2.0/100,   40.0/100,  26.0/100,  7.0/100,   9.0/100,   26.0/100,  1.0/100,   1.0/100,   1.0/100,   60.0/100 ],
        'person_x_max': [528.0/100, 201.0/100, 223.0/100, 261.0/100, 290.0/100, 228.0/100, 426.0/100, 400.0/100, 448.0/100, 304.0/100, 266.0/100],
        'person_y_max': [224.0/100, 211.0/100, 256.0/100, 256.0/100, 224.0/100, 256.0/100, 224.0/100, 256.0/100, 256.0/100, 256.0/100, 190.0/100]
    }

    predictions = classifier.predict(
        input_fn=lambda:shot_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(shot_data.CLASS[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
