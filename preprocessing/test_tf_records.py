import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import io
from PIL import Image

tf.compat.v1.enable_eager_execution()

tf.disable_v2_behavior()

tfrecord_path = r"F:\code_pfe_all\IP102_DATASET\dataset\final_models\tfrecords\train.record"

def visualize_tfrecord(tfrecord_path, num_images=5):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

    # Define the feature description for decoding
    feature_description = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/source_id': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/class/text': tf.VarLenFeature(tf.string),
        'image/object/class/label': tf.VarLenFeature(tf.int64),
    }

    def _parse_function(example_proto):
        return tf.parse_single_example(example_proto, feature_description)

    parsed_dataset = raw_dataset.map(_parse_function)

    plt.figure(figsize=(15, 10))
    for i, parsed_record in enumerate(parsed_dataset.take(num_images)):
        image_encoded = parsed_record['image/encoded'].numpy()
        image = Image.open(io.BytesIO(image_encoded))

        plt.subplot(1, num_images, i + 1)
        plt.imshow(image)
        plt.title(parsed_record['image/filename'].numpy().decode())
        plt.axis('off')

    plt.show()

visualize_tfrecord(tfrecord_path)
