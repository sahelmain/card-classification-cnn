import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (200, 200)
BATCH_SIZE = 64  # Larger batch size for better GPU utilization

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

def get_generators():
    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    
    valid_generator = valid_datagen.flow_from_directory(
        'data/valid',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    
    test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    
    return train_generator, valid_generator, test_generator 