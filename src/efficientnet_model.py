import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_preprocessing import get_generators
from gpu_config import setup_gpu

# Setup GPU acceleration
setup_gpu()

NUM_CLASSES = 53

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_gen, valid_gen, test_gen = get_generators()

# Add callbacks for better training
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-7)
]

history = model.fit(train_gen, epochs=10, validation_data=valid_gen, callbacks=callbacks)

# Fine-tune
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tuning with callbacks
fine_callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-8)
]

history_fine = model.fit(train_gen, epochs=10, validation_data=valid_gen, callbacks=fine_callbacks)

model.evaluate(test_gen)

model.save('results/efficientnet_b0.h5') 