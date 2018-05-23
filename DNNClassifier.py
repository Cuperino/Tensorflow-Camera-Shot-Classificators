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
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = shot_data.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build i hidden layers of n nodes each: [n(1), n(2), [...], n(i)]
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[90, 90],
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
    expected = ['CU', 'LS', 'MS', 'MCU', 'MS', 'MLS', 'MCU', 'CU', 'ECU', 'MCU', 'ELS']
    predict_x = {
        'frame_height': [224.0, 256.0, 256.0, 256.0, 224.0, 256.0, 224.0, 256.0, 256.0, 256.0, 224.0],
        'face_x_min':   [254.0, 154.0, 108.0, 142.0, 197.0, 192.0, 252.0, 168.0, 1.0,   182.0, 246.0],
        'face_y_min':   [1.0,   25.0,  88.0,  51.0,  25.0,  21.0,  40.0,  35.0,  1.0,   19.0,  62.0 ],
        'face_x_max':   [427.0, 167.0, 155.0, 204.0, 244.0, 216.0, 303.0, 270.0, 434.0, 235.0, 258.0],
        'face_y_max':   [192.0, 47.0,  125.0, 116.0, 90.0,  59.0,  114.0, 218.0, 256.0, 113.0, 75.0 ],
        'person_x_min': [186.0, 144.0, 61.0,  58.0,  135.0, 165.0, 196.0, 145.0, 1.0,   108.0, 232.0],
        'person_y_min': [1.0,   20.0,  40.0,  26.0,  7.0,   9.0,   26.0,  1.0,   1.0,   1.0,   60.0 ],
        'person_x_max': [528.0, 201.0, 223.0, 261.0, 290.0, 228.0, 426.0, 400.0, 448.0, 304.0, 266.0],
        'person_y_max': [224.0, 211.0, 256.0, 256.0, 224.0, 256.0, 224.0, 256.0, 256.0, 256.0, 190.0]
    }

    # 224.0,254.0, 1.0,427.0,192.0,186.0, 1.0,528.0,224.0, 5    CU
    # 256.0,154.0,25.0,167.0, 47.0,144.0,20.0,201.0,211.0, 1    LS
    # 256.0,108.0,88.0,155.0,125.0, 61.0,40.0,223.0,256.0, 3    MS
    # 256.0,142.0,51.0,204.0,116.0, 58.0,26.0,261.0,256.0, 4    MCU
    # 224.0,197.0,25.0,244.0, 90.0,135.0, 7.0,290.0,224.0, 3    MS
    # 256.0,192.0,21.0,216.0, 59.0,165.0, 9.0,228.0,256.0, 2    MLS
    # 224.0,252.0,40.0,303.0,114.0,196.0,26.0,426.0,224.0, 4    MCU
    # 256.0,168.0,35.0,270.0,218.0,145.0, 1.0,400.0,256.0, 5    CU
    # 256.0,  1.0, 1.0,434.0,256.0,  1.0, 1.0,448.0,256.0, 6    ECU
    # 256.0,182.0,19.0,235.0,113.0,108.0, 1.0,304.0,256.0, 4    MCU
    # 224.0,246.0,62.0,258.0, 75.0,232.0,60.0,266.0,190.0, 0    ELS

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
