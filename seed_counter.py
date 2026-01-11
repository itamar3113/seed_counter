"""
Module for counting seeds on disc images.
This module processes disc images to detect and count seeds.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class SeedCounter:
    """Count seeds on white disc images."""
    
    def __init__(self, min_seed_area: int = 10, max_seed_area: int = 1000):
        """
        Initialize the SeedCounter.
        
        Args:
            min_seed_area: Minimum area of a valid seed in pixels
            max_seed_area: Maximum area of a valid seed in pixels
        """
        self.min_seed_area = min_seed_area
        self.max_seed_area = max_seed_area
    
    def count_seeds(self, disc_image: np.ndarray, visualize: bool = False) -> Tuple[int, Optional[np.ndarray]]:
        """
        Count seeds on a disc image.
        
        Args:
            disc_image: Image of a disc with seeds (numpy array)
            visualize: If True, return an annotated image showing detected seeds
            
        Returns:
            Tuple of (seed_count, annotated_image)
            annotated_image is None if visualize=False
        """
        if disc_image is None or disc_image.size == 0:
            return 0, None
        
        # Convert to grayscale if needed
        if len(disc_image.shape) == 3:
            gray = cv2.cvtColor(disc_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = disc_image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Seeds are typically darker than the white disc
        # Use adaptive threshold to detect seeds
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Find contours (potential seeds)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        seed_count = 0
        seed_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area to get only seed-sized objects
            if self.min_seed_area <= area <= self.max_seed_area:
                seed_count += 1
                seed_contours.append(contour)
        
        annotated_image = None
        if visualize:
            # Create a copy of the original image for annotation
            if len(disc_image.shape) == 3:
                annotated_image = disc_image.copy()
            else:
                annotated_image = cv2.cvtColor(disc_image, cv2.COLOR_GRAY2BGR)
            
            # Draw contours around detected seeds
            cv2.drawContours(annotated_image, seed_contours, -1, (0, 255, 0), 2)
            
            # Add text showing the count
            cv2.putText(
                annotated_image,
                f"Seeds: {seed_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        
        return seed_count, annotated_image
    
    def count_seeds_from_image_path(self, image_path: str, visualize: bool = False) -> Tuple[int, Optional[np.ndarray]]:
        """
        Count seeds from an image file path.
        
        Args:
            image_path: Path to the disc image
            visualize: If True, return an annotated image showing detected seeds
            
        Returns:
            Tuple of (seed_count, annotated_image)
            annotated_image is None if visualize=False
        """
        disc_image = cv2.imread(image_path)
        if disc_image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        return self.count_seeds(disc_image, visualize)
    
    def count_seeds_blob_detection(self, disc_image: np.ndarray, visualize: bool = False) -> Tuple[int, Optional[np.ndarray]]:
        """
        Count seeds using blob detection.
        This is an alternative method that may work better for certain seed types.
        
        Args:
            disc_image: Image of a disc with seeds (numpy array)
            visualize: If True, return an annotated image showing detected seeds
            
        Returns:
            Tuple of (seed_count, annotated_image)
            annotated_image is None if visualize=False
        """
        if disc_image is None or disc_image.size == 0:
            return 0, None
        
        # Convert to grayscale if needed
        if len(disc_image.shape) == 3:
            gray = cv2.cvtColor(disc_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = disc_image.copy()
        
        # Invert the image (seeds should be bright spots on dark background for blob detection)
        inverted = cv2.bitwise_not(gray)
        
        # Set up the SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()
        
        # Filter by area
        params.filterByArea = True
        params.minArea = self.min_seed_area
        params.maxArea = self.max_seed_area
        
        # Filter by circularity (seeds are often somewhat circular)
        params.filterByCircularity = True
        params.minCircularity = 0.3
        
        # Filter by convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5
        
        # Filter by inertia (roundness)
        params.filterByInertia = True
        params.minInertiaRatio = 0.2
        
        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(inverted)
        
        seed_count = len(keypoints)
        
        annotated_image = None
        if visualize:
            # Create a copy of the original image for annotation
            if len(disc_image.shape) == 3:
                annotated_image = disc_image.copy()
            else:
                annotated_image = cv2.cvtColor(disc_image, cv2.COLOR_GRAY2BGR)
            
            # Draw detected blobs as red circles
            annotated_image = cv2.drawKeypoints(
                annotated_image,
                keypoints,
                np.array([]),
                (0, 0, 255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            
            # Add text showing the count
            cv2.putText(
                annotated_image,
                f"Seeds: {seed_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        
        return seed_count, annotated_image
