import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from data_preprocessing import get_generators
from gpu_config import setup_gpu
import time

# Setup GPU/CPU acceleration
setup_gpu()

print("🚀 Starting Quick Demo Training...")

NUM_CLASSES = 53

def create_lightweight_cnn():
    """Smaller, faster CNN for quick demo"""
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
        MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def create_mobilenet_model():
    """Lightweight transfer learning with MobileNetV2"""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
    
    # Freeze base model
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Load data
print("📁 Loading dataset...")
train_gen, valid_gen, test_gen = get_generators()

# Early stopping for efficiency
early_stop = EarlyStopping(patience=3, restore_best_weights=True, verbose=1)

print("\n🏗️  Training Lightweight CNN...")
start_time = time.time()

# Lightweight CNN
lightweight_cnn = create_lightweight_cnn()
lightweight_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Model Summary:")
lightweight_cnn.summary()

# Train for fewer epochs
history1 = lightweight_cnn.fit(
    train_gen, 
    epochs=5,  # Reduced epochs for speed
    validation_data=valid_gen,
    callbacks=[early_stop],
    verbose=1
)

cnn_time = time.time() - start_time
print(f"⏱️  Lightweight CNN training time: {cnn_time:.1f} seconds")

# Evaluate
print("\n📊 Evaluating Lightweight CNN...")
cnn_results = lightweight_cnn.evaluate(test_gen, verbose=0)
print(f"Lightweight CNN - Test Accuracy: {cnn_results[1]:.3f}")

# Save model
lightweight_cnn.save('results/lightweight_cnn.h5')

print("\n🚀 Training MobileNetV2 (Transfer Learning)...")
start_time = time.time()

# MobileNetV2 with transfer learning
mobilenet_model = create_mobilenet_model()
mobilenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train for fewer epochs
history2 = mobilenet_model.fit(
    train_gen,
    epochs=3,  # Very few epochs for demo
    validation_data=valid_gen,
    callbacks=[early_stop],
    verbose=1
)

mobilenet_time = time.time() - start_time
print(f"⏱️  MobileNetV2 training time: {mobilenet_time:.1f} seconds")

# Evaluate
print("\n📊 Evaluating MobileNetV2...")
mobilenet_results = mobilenet_model.evaluate(test_gen, verbose=0)
print(f"MobileNetV2 - Test Accuracy: {mobilenet_results[1]:.3f}")

# Save model
mobilenet_model.save('results/mobilenet_v2.h5')

print("\n" + "="*50)
print("🎉 QUICK DEMO RESULTS SUMMARY")
print("="*50)
print(f"Lightweight CNN:")
print(f"  ⏱️  Training Time: {cnn_time:.1f}s")
print(f"  🎯 Test Accuracy: {cnn_results[1]:.3f}")
print(f"\nMobileNetV2 (Transfer Learning):")
print(f"  ⏱️  Training Time: {mobilenet_time:.1f}s") 
print(f"  🎯 Test Accuracy: {mobilenet_results[1]:.3f}")
print("\n💡 This demonstrates both custom CNN and transfer learning approaches!")
print("   For production use, train with more epochs for better accuracy.") 