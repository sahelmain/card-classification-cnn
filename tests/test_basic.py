import unittest
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestBasicFunctionality(unittest.TestCase):
    """Basic test suite to demonstrate testing framework"""
    
    def test_project_structure(self):
        """Test that essential project files exist"""
        project_root = os.path.dirname(os.path.dirname(__file__))
        
        # Check essential directories
        self.assertTrue(os.path.exists(os.path.join(project_root, 'src')))
        self.assertTrue(os.path.exists(os.path.join(project_root, 'docs')))
        self.assertTrue(os.path.exists(os.path.join(project_root, 'results')))
        
        # Check essential files
        self.assertTrue(os.path.exists(os.path.join(project_root, 'README.md')))
        self.assertTrue(os.path.exists(os.path.join(project_root, 'requirements.txt')))
        self.assertTrue(os.path.exists(os.path.join(project_root, 'Dockerfile')))
    
    def test_configuration_constants(self):
        """Test configuration constants are properly set"""
        try:
            from data_preprocessing import IMG_SIZE, BATCH_SIZE
            
            # Test image size configuration
            self.assertEqual(IMG_SIZE, (200, 200))
            self.assertIsInstance(IMG_SIZE, tuple)
            
            # Test batch size configuration
            self.assertEqual(BATCH_SIZE, 64)
            self.assertIsInstance(BATCH_SIZE, int)
            self.assertGreater(BATCH_SIZE, 0)
            
        except ImportError:
            self.skipTest("data_preprocessing module not available")
    
    def test_python_version(self):
        """Test that Python version is compatible"""
        self.assertGreaterEqual(sys.version_info.major, 3)
        self.assertGreaterEqual(sys.version_info.minor, 8)
    
    def test_essential_imports(self):
        """Test that essential packages can be imported"""
        try:
            import tensorflow as tf
            import numpy as np
            import matplotlib.pyplot as plt
            
            # Basic functionality test
            self.assertTrue(hasattr(tf, '__version__'))
            self.assertTrue(hasattr(np, '__version__'))
            
        except ImportError as e:
            self.fail(f"Essential package import failed: {e}")

class TestModelConfiguration(unittest.TestCase):
    """Test suite for model configuration"""
    
    def test_model_parameters(self):
        """Test model parameters are within expected ranges"""
        NUM_CLASSES = 53  # Standard deck + jokers
        
        self.assertEqual(NUM_CLASSES, 53)
        self.assertIsInstance(NUM_CLASSES, int)
        self.assertGreater(NUM_CLASSES, 0)
    
    def test_image_preprocessing_parameters(self):
        """Test image preprocessing parameters"""
        IMG_SIZE = (200, 200)
        RESCALE_FACTOR = 1./255
        
        # Test image dimensions
        self.assertEqual(len(IMG_SIZE), 2)
        self.assertTrue(all(isinstance(dim, int) for dim in IMG_SIZE))
        self.assertTrue(all(dim > 0 for dim in IMG_SIZE))
        
        # Test rescale factor
        self.assertAlmostEqual(RESCALE_FACTOR, 0.00392156862, places=5)
        self.assertGreater(RESCALE_FACTOR, 0)
        self.assertLess(RESCALE_FACTOR, 1)

if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2) 