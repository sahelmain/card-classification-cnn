import tensorflow as tf
from data_preprocessing import get_generators

_, _, test_gen = get_generators()

custom_model = tf.keras.models.load_model('results/custom_cnn.h5')
print('Custom CNN Evaluation:')
custom_model.evaluate(test_gen)

effnet_model = tf.keras.models.load_model('results/efficientnet_b0.h5')
print('EfficientNet-B0 Evaluation:')
effnet_model.evaluate(test_gen) 