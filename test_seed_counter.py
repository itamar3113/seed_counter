"""
Unit tests for the seed counter modules.
Tests basic functionality without requiring actual images.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestDiscExtractor(unittest.TestCase):
    """Test cases for DiscExtractor class."""
    
    def test_init(self):
        """Test DiscExtractor initialization."""
        # Import here to avoid issues if cv2 is not installed
        try:
            from disc_extractor import DiscExtractor
            
            # Test default initialization
            extractor = DiscExtractor()
            self.assertEqual(extractor.min_disc_area, 5000)
            self.assertEqual(extractor.max_disc_area, 500000)
            self.assertEqual(extractor.debug, False)
            
            # Test custom initialization
            extractor = DiscExtractor(min_disc_area=1000, max_disc_area=100000, debug=True)
            self.assertEqual(extractor.min_disc_area, 1000)
            self.assertEqual(extractor.max_disc_area, 100000)
            self.assertEqual(extractor.debug, True)
            
        except ImportError:
            self.skipTest("OpenCV not installed, skipping test")
    
    def test_extract_discs_invalid_path(self):
        """Test that extract_discs raises error for invalid image path."""
        try:
            from disc_extractor import DiscExtractor
            
            extractor = DiscExtractor()
            with self.assertRaises(ValueError):
                extractor.extract_discs("nonexistent_image.jpg")
                
        except ImportError:
            self.skipTest("OpenCV not installed, skipping test")


class TestSeedCounter(unittest.TestCase):
    """Test cases for SeedCounter class."""
    
    def test_init(self):
        """Test SeedCounter initialization."""
        try:
            from seed_counter import SeedCounter
            
            # Test default initialization
            counter = SeedCounter()
            self.assertEqual(counter.min_seed_area, 10)
            self.assertEqual(counter.max_seed_area, 1000)
            
            # Test custom initialization
            counter = SeedCounter(min_seed_area=5, max_seed_area=500)
            self.assertEqual(counter.min_seed_area, 5)
            self.assertEqual(counter.max_seed_area, 500)
            
        except ImportError:
            self.skipTest("OpenCV not installed, skipping test")
    
    def test_count_seeds_empty_image(self):
        """Test that count_seeds handles empty images gracefully."""
        try:
            from seed_counter import SeedCounter
            
            counter = SeedCounter()
            
            # Test with None
            count, annotated = counter.count_seeds(None)
            self.assertEqual(count, 0)
            self.assertIsNone(annotated)
            
            # Test with empty array
            empty_array = np.array([])
            count, annotated = counter.count_seeds(empty_array)
            self.assertEqual(count, 0)
            self.assertIsNone(annotated)
            
        except ImportError:
            self.skipTest("OpenCV not installed, skipping test")
    
    def test_count_seeds_from_invalid_path(self):
        """Test that count_seeds_from_image_path raises error for invalid path."""
        try:
            from seed_counter import SeedCounter
            
            counter = SeedCounter()
            with self.assertRaises(ValueError):
                counter.count_seeds_from_image_path("nonexistent_image.jpg")
                
        except ImportError:
            self.skipTest("OpenCV not installed, skipping test")


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_modules_can_be_imported(self):
        """Test that all modules can be imported (if dependencies are available)."""
        try:
            from disc_extractor import DiscExtractor
            from seed_counter import SeedCounter
            
            # If we get here, imports succeeded
            self.assertTrue(True)
            
        except ImportError as e:
            # This is okay if cv2 is not installed
            if "cv2" in str(e):
                self.skipTest("OpenCV not installed, skipping test")
            else:
                # Re-raise if it's a different import error
                raise


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestDiscExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestSeedCounter))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
