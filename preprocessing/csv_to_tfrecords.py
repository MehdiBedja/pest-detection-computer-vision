from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple

def class_text_to_int(row_label):
    class_map = {
        '0': 'IP014',
        '1': 'IP015',
        '2': 'IP016',
        '3': 'IP022',
        '4': 'IP024',
        '5': 'IP040',
        '6': 'IP046',
        '7': 'IP049',
        '8': 'IP051',
        '9': 'IP052',
        '10': 'IP071',
        '11': 'IP087',
        '12': 'IP102'
    }
    return class_map.get(row_label, None)

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(float(row['xmin']))
        xmaxs.append(float(row['xmax']))
        ymins.append(float(row['ymin']))
        ymaxs.append(float(row['ymax']))
        class_name = class_text_to_int(str(row['class']))
        if class_name:
            classes_text.append(class_name.encode('utf8'))
            classes.append(int(row['class']) + 1)  # Shift class ids by +1

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):
    csv_input = r"F:\code_pfe_all\code_dataset_models\generating_dataset_testing_running_modelV2\send\test_annotations.csv"
    output_path = r"F:\code_pfe_all\IP102_DATASET\dataset\final_models\tfrecords\test.record"
    image_dir = r"F:\code_pfe_all\IP102_DATASET\dataset\final_models\yolo_dataset\dataset_Yolo_version_13pestTypes\test\images"

    writer = tf.python_io.TFRecordWriter(output_path)
    path = os.path.join(image_dir)
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print(f'Successfully created the TFRecord file at {output_path}')

if __name__ == '__main__':
    tf.app.run()
