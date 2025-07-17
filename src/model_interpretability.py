import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os

class ModelInterpretability:
    """
    Comprehensive model interpretability and explainability toolkit.
    Implements Grad-CAM and feature visualization techniques.
    """
    
    def __init__(self, model, class_names=None):
        self.model = model
        self.class_names = class_names or [f'Class_{i}' for i in range(53)]
        self.results_dir = 'results/interpretability'
        os.makedirs(self.results_dir, exist_ok=True)
    
    def grad_cam(self, img_array, class_index=None, layer_name=None):
        """
        Generate Grad-CAM heatmap for model predictions
        """
        # Find the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(self.model.layers):
                if len(layer.output_shape) == 4:  # Conv layer
                    layer_name = layer.name
                    break
        
        if layer_name is None:
            raise ValueError("No convolutional layer found in model")
        
        # Create grad model
        grad_model = tf.keras.models.Model(
            [self.model.inputs], 
            [self.model.get_layer(layer_name).output, self.model.output]
        )
        
        # Calculate gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            class_output = predictions[:, class_index]
        
        # Get gradients
        grads = tape.gradient(class_output, conv_outputs)
        
        # Pool gradients over spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def overlay_heatmap(self, img, heatmap, alpha=0.4):
        """Overlay heatmap on original image"""
        # Resize heatmap to match image
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Apply colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert to RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        overlayed = heatmap * alpha + img * (1 - alpha)
        return overlayed.astype(np.uint8)
    
    def explain_prediction(self, img_array, save_path=None):
        """Generate comprehensive explanation for a single prediction"""
        # Get prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Generate Grad-CAM
        heatmap = self.grad_cam(img_array, predicted_class)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        img_display = img_array[0]
        axes[0, 0].imshow(img_display)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Heatmap
        axes[0, 1].imshow(heatmap, cmap='jet')
        axes[0, 1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Overlay
        img_np = (img_display * 255).astype(np.uint8)
        overlayed = self.overlay_heatmap(img_np, heatmap)
        axes[1, 0].imshow(overlayed)
        axes[1, 0].set_title('Grad-CAM Overlay', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Top predictions
        top_indices = np.argsort(predictions[0])[-5:][::-1]
        top_confidences = predictions[0][top_indices]
        top_classes = [self.class_names[i] for i in top_indices]
        
        axes[1, 1].barh(range(5), top_confidences)
        axes[1, 1].set_yticks(range(5))
        axes[1, 1].set_yticklabels(top_classes)
        axes[1, 1].set_xlabel('Confidence')
        axes[1, 1].set_title('Top 5 Predictions', fontsize=14, fontweight='bold')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Explanation saved to {save_path}")
        
        plt.show()
        
        # Print summary
        print(f"\n🔍 Prediction Explanation")
        print(f"{'='*50}")
        print(f"Predicted Class: {self.class_names[predicted_class]}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'heatmap': heatmap
        }
    
    def analyze_feature_importance(self, img_array, grid_size=4):
        """Analyze feature importance by occluding different regions"""
        original_pred = self.model.predict(img_array, verbose=0)[0]
        original_confidence = np.max(original_pred)
        
        importance_scores = []
        h, w = img_array.shape[1:3]
        step_h, step_w = h // grid_size, w // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Create occluded image
                occluded_img = img_array.copy()
                start_h, end_h = i * step_h, (i + 1) * step_h
                start_w, end_w = j * step_w, (j + 1) * step_w
                occluded_img[0, start_h:end_h, start_w:end_w, :] = 0
                
                # Get prediction
                occluded_pred = self.model.predict(occluded_img, verbose=0)[0]
                occluded_confidence = np.max(occluded_pred)
                
                # Calculate importance as confidence drop
                importance = original_confidence - occluded_confidence
                importance_scores.append(importance)
        
        return importance_scores
    
    def visualize_filters(self, layer_name=None, num_filters=16):
        """Visualize convolutional filters"""
        if layer_name is None:
            # Find first conv layer
            for layer in self.model.layers:
                if len(layer.output_shape) == 4:
                    layer_name = layer.name
                    break
        
        layer = self.model.get_layer(layer_name)
        weights = layer.get_weights()[0]  # Get filter weights
        
        # Normalize weights
        weights = (weights - weights.min()) / (weights.max() - weights.min())
        
        # Plot filters
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i in range(min(num_filters, 16)):
            row, col = i // 4, i % 4
            filter_img = weights[:, :, 0, i]  # First channel, i-th filter
            axes[row, col].imshow(filter_img, cmap='viridis')
            axes[row, col].set_title(f'Filter {i}')
            axes[row, col].axis('off')
        
        plt.suptitle(f'Filters from {layer_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/filters_{layer_name}.png', dpi=300)
        plt.show()

def demonstrate_interpretability():
    """Demonstrate model interpretability capabilities"""
    print("🔍 Model Interpretability Demonstration")
    print("=" * 50)
    
    # Create dummy model for demonstration
    dummy_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(200, 200, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(53, activation='softmax')
    ])
    
    print("✅ Grad-CAM analysis capability")
    print("✅ Feature visualization")
    print("✅ Filter visualization") 
    print("✅ Occlusion-based importance")
    
    print("\n🎯 Key Features:")
    print("• Visual explanations with Grad-CAM")
    print("• Feature importance analysis")
    print("• Filter visualization")
    print("• Confidence analysis")

if __name__ == "__main__":
    demonstrate_interpretability() 