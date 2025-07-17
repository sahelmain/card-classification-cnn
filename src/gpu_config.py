import tensorflow as tf

def setup_gpu():
    """Configure GPU settings for optimal performance"""
    # Check for Apple Silicon MPS (Metal Performance Shaders) first
    if tf.config.list_physical_devices('GPU'):
        try:
            # For Apple Silicon Macs
            if hasattr(tf.config.experimental, 'set_device_policy'):
                tf.config.experimental.set_device_policy('GPU', 'explicit')
            print("✅ Apple Metal GPU acceleration enabled!")
            return True
        except Exception as e:
            print(f"❌ Metal GPU setup error: {e}")
    
    # Check for traditional CUDA GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ CUDA GPU acceleration enabled! Found {len(gpus)} GPU(s)")
            
            # Enable mixed precision for faster training
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("✅ Mixed precision training enabled")
            
        except RuntimeError as e:
            print(f"❌ GPU setup error: {e}")
        return True
    else:
        # Check if MPS (Metal Performance Shaders) is available for Apple Silicon
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([1.0, 2.0, 3.0])
                b = tf.constant([4.0, 5.0, 6.0])
                c = tf.add(a, b)
            print("✅ Apple Metal GPU acceleration available!")
            return True
        except:
            pass
        
        print("❌ No GPU acceleration available, using optimized CPU")
        # Enable CPU optimizations
        tf.config.threading.set_inter_op_parallelism_threads(0)
        tf.config.threading.set_intra_op_parallelism_threads(0)
        print("✅ CPU optimizations enabled")
    
    return False

if __name__ == "__main__":
    setup_gpu() 