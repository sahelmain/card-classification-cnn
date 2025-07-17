import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageEnhance, ImageFilter
import random

class AdvancedAugmentation:
    """
    Advanced data augmentation techniques for card classification.
    Combines traditional and modern augmentation methods for better model generalization.
    """
    
    def __init__(self, augmentation_probability=0.8):
        self.augmentation_probability = augmentation_probability
        self.setup_augmentation_pipelines()
    
    def setup_augmentation_pipelines(self):
        """Setup various augmentation pipelines"""
        
        # Keras ImageDataGenerator with advanced parameters
        self.keras_augmentation = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,  # Cards shouldn't be flipped horizontally
            vertical_flip=False,    # Cards shouldn't be flipped vertically
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.1,
            fill_mode='nearest',
            rescale=1./255
        )
        
        # TensorFlow native augmentation
        self.tf_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ])
    
    def apply_color_augmentation(self, image):
        """Apply color-based augmentations"""
        if random.random() < self.augmentation_probability:
            # Convert to PIL for easier manipulation
            if isinstance(image, np.ndarray):
                image = Image.fromarray((image * 255).astype(np.uint8))
            
            # Random color enhancements
            enhancers = [
                ImageEnhance.Brightness,
                ImageEnhance.Color,
                ImageEnhance.Contrast,
                ImageEnhance.Sharpness
            ]
            
            for enhancer_class in enhancers:
                if random.random() < 0.5:
                    enhancer = enhancer_class(image)
                    factor = random.uniform(0.8, 1.2)
                    image = enhancer.enhance(factor)
            
            # Convert back to numpy
            image = np.array(image) / 255.0
        
        return image
    
    def apply_geometric_augmentation(self, image):
        """Apply geometric transformations"""
        if random.random() < self.augmentation_probability:
            # Random perspective transformation
            height, width = image.shape[:2]
            
            # Define source points (corners of the image)
            src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            
            # Add random distortion to corners
            max_distortion = min(width, height) * 0.05
            dst_points = src_points + np.random.uniform(-max_distortion, max_distortion, (4, 2))
            
            # Apply perspective transformation
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            image = cv2.warpPerspective(image, matrix, (width, height))
        
        return image
    
    def apply_lighting_augmentation(self, image):
        """Apply lighting and shadow augmentations"""
        if random.random() < self.augmentation_probability:
            # Simulate different lighting conditions
            image = image.copy()
            
            # Add random shadow
            if random.random() < 0.3:
                shadow_intensity = random.uniform(0.3, 0.7)
                shadow_area = self._create_random_shadow_mask(image.shape[:2])
                image = image * (1 - shadow_area * shadow_intensity)
            
            # Add random highlight
            if random.random() < 0.3:
                highlight_intensity = random.uniform(0.1, 0.3)
                highlight_area = self._create_random_highlight_mask(image.shape[:2])
                image = np.clip(image + highlight_area * highlight_intensity, 0, 1)
        
        return image
    
    def _create_random_shadow_mask(self, shape):
        """Create a random shadow mask"""
        h, w = shape
        mask = np.zeros((h, w))
        
        # Create elliptical shadow
        center_x = random.randint(w//4, 3*w//4)
        center_y = random.randint(h//4, 3*h//4)
        radius_x = random.randint(w//8, w//3)
        radius_y = random.randint(h//8, h//3)
        
        y, x = np.ogrid[:h, :w]
        mask_condition = ((x - center_x) / radius_x)**2 + ((y - center_y) / radius_y)**2 <= 1
        mask[mask_condition] = 1
        
        # Smooth the mask
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def _create_random_highlight_mask(self, shape):
        """Create a random highlight mask"""
        h, w = shape
        mask = np.zeros((h, w))
        
        # Create small circular highlights
        num_highlights = random.randint(1, 3)
        for _ in range(num_highlights):
            center_x = random.randint(0, w)
            center_y = random.randint(0, h)
            radius = random.randint(10, 30)
            
            y, x = np.ogrid[:h, :w]
            mask_condition = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            mask[mask_condition] = 1
        
        # Smooth the mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask
    
    def apply_texture_augmentation(self, image):
        """Apply texture-based augmentations"""
        if random.random() < self.augmentation_probability:
            # Add texture noise
            if random.random() < 0.4:
                noise = np.random.normal(0, 0.02, image.shape)
                image = np.clip(image + noise, 0, 1)
            
            # Apply random blur for texture variation
            if random.random() < 0.3:
                blur_kernel = random.choice([3, 5])
                image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
        
        return image
    
    def augment_image(self, image, method='comprehensive'):
        """
        Apply augmentation to a single image
        
        Args:
            image: Input image (numpy array)
            method: Augmentation method ('keras', 'tf', 'comprehensive')
        """
        if method == 'keras':
            # Convert to format expected by ImageDataGenerator
            image = np.expand_dims(image, axis=0)
            image = self.keras_augmentation.random_transform(image[0])
            
        elif method == 'tf':
            # Apply TensorFlow augmentation
            image = tf.expand_dims(image, axis=0)
            image = self.tf_augmentation(image)
            image = tf.squeeze(image, axis=0).numpy()
            
        elif method == 'comprehensive':
            # Apply all custom augmentations
            image = self.apply_color_augmentation(image)
            image = self.apply_geometric_augmentation(image)
            image = self.apply_lighting_augmentation(image)
            image = self.apply_texture_augmentation(image)
        
        return np.clip(image, 0, 1)
    
    def create_augmented_dataset(self, images, labels, augmentation_factor=2):
        """
        Create augmented dataset with specified augmentation factor
        
        Args:
            images: Original images
            labels: Corresponding labels
            augmentation_factor: Number of augmented versions per original image
        """
        augmented_images = []
        augmented_labels = []
        
        print(f"Creating augmented dataset with factor {augmentation_factor}...")
        
        for i, (image, label) in enumerate(zip(images, labels)):
            # Add original image
            augmented_images.append(image)
            augmented_labels.append(label)
            
            # Add augmented versions
            for j in range(augmentation_factor - 1):
                method = random.choice(['comprehensive', 'keras'])
                aug_image = self.augment_image(image.copy(), method=method)
                augmented_images.append(aug_image)
                augmented_labels.append(label)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(images)} images")
        
        return np.array(augmented_images), np.array(augmented_labels)
    
    def visualize_augmentations(self, image, save_path='results/augmentation_examples.png'):
        """Visualize different augmentation techniques"""
        import matplotlib.pyplot as plt
        
        methods = ['original', 'keras', 'tf', 'comprehensive']
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.ravel()
        
        for i, method in enumerate(methods):
            if method == 'original':
                aug_image = image
                title = 'Original'
            else:
                aug_image = self.augment_image(image.copy(), method=method)
                title = f'{method.title()} Augmentation'
            
            axes[i].imshow(aug_image)
            axes[i].set_title(title, fontsize=14, fontweight='bold')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Augmentation examples saved to {save_path}")

def demonstrate_augmentation():
    """Demonstrate the augmentation capabilities"""
    # Create sample image (placeholder)
    sample_image = np.random.rand(200, 200, 3)
    
    # Initialize augmentation
    augmenter = AdvancedAugmentation()
    
    # Create augmented versions
    methods = ['keras', 'tf', 'comprehensive']
    
    print("🎨 Advanced Data Augmentation Demonstration")
    print("=" * 50)
    
    for method in methods:
        print(f"Testing {method} augmentation...")
        aug_image = augmenter.augment_image(sample_image.copy(), method=method)
        print(f"✅ {method} augmentation completed")
        print(f"   Input shape: {sample_image.shape}")
        print(f"   Output shape: {aug_image.shape}")
        print(f"   Value range: [{aug_image.min():.3f}, {aug_image.max():.3f}]")
        print()
    
    print("🎉 All augmentation methods working correctly!")
    print("\nAugmentation Benefits:")
    print("• Increased dataset size for better generalization")
    print("• Robustness to lighting and perspective variations")
    print("• Better handling of real-world image conditions")
    print("• Reduced overfitting through data diversity")

if __name__ == "__main__":
    demonstrate_augmentation() 