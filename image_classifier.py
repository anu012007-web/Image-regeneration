import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class ImageClassifier:
    """Image type classification module"""
    
    def __init__(self):
        self.collage_indicators = []
    
    def analyze(self, image):
        """
        Analyze if image is a collage or single image
        
        Args:
            image: numpy array image (BGR format)
        
        Returns:
            Dictionary with image type analysis
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            h, w = gray.shape
            
            # Perform various analyses
            line_analysis = self._detect_straight_lines(gray, h, w)
            region_analysis = self._detect_color_regions(image)
            border_analysis = self._detect_borders(gray, h, w)
            texture_analysis = self._analyze_texture(gray)
            
            # Calculate collage probability
            collage_score = self._calculate_collage_score(
                line_analysis, region_analysis, border_analysis, texture_analysis
            )
            
            # Determine if collage
            is_collage = bool(collage_score > 0.5)
            confidence = collage_score * 100
            
            # Determine specific type
            specific_type = self._determine_specific_type(
                is_collage, line_analysis, region_analysis, border_analysis
            )
            
            return {
                'is_collage': is_collage,
                'confidence': round(confidence, 1),
                'type': specific_type,
                'details': {
                    'has_straight_lines': line_analysis['has_lines'],
                    'line_count': line_analysis['line_count'],
                    'has_multiple_regions': region_analysis['has_multiple_regions'],
                    'region_count': region_analysis['region_count'],
                    'has_uniform_border': border_analysis['has_uniform_border'],
                    'texture_variance': round(texture_analysis['variance'], 2)
                },
                'recommendations': self._get_recommendations(is_collage, specific_type)
            }
        
        except Exception as e:
            logger.error(f"Image classification error: {e}")
            return {
                'is_collage': False,
                'confidence': 0,
                'type': 'Unknown',
                'details': {},
                'recommendations': ['Unable to classify image type']
            }
    
    def _detect_straight_lines(self, gray, height, width):
        """Detect straight lines that might indicate collage borders"""
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=100,
            minLineLength=min(height, width)//4, maxLineGap=10
        )
        
        has_lines = False
        line_count = 0
        horizontal_lines = 0
        vertical_lines = 0
        
        if lines is not None:
            line_count = len(lines)
            for line in lines[:20]:  # Check first 20 lines
                x1, y1, x2, y2 = line[0]
                # Check if line is horizontal or vertical
                if abs(x2 - x1) < 5:  # Vertical line
                    vertical_lines += 1
                    has_lines = True
                elif abs(y2 - y1) < 5:  # Horizontal line
                    horizontal_lines += 1
                    has_lines = True
        
        return {
            'has_lines': has_lines,
            'line_count': line_count,
            'horizontal_lines': horizontal_lines,
            'vertical_lines': vertical_lines
        }
    
    def _detect_color_regions(self, image):
        """Detect distinct color regions using clustering"""
        try:
            # Reshape image to pixel list
            pixels = image.reshape(-1, 3)
            
            # Sample pixels for performance
            sample_size = min(5000, len(pixels))
            indices = np.random.choice(len(pixels), sample_size, replace=False)
            sampled_pixels = pixels[indices]
            
            # Apply K-means clustering
            n_clusters = min(8, len(np.unique(sampled_pixels, axis=0)))
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(sampled_pixels)
                labels = kmeans.labels_
                
                # Count pixels per cluster
                cluster_counts = Counter(labels)
                
                # Count significant clusters (>10% of sample)
                significant_clusters = sum(1 for count in cluster_counts.values() 
                                          if count / sample_size > 0.1)
                
                return {
                    'has_multiple_regions': significant_clusters >= 3,
                    'region_count': significant_clusters,
                    'total_clusters': n_clusters
                }
            else:
                return {
                    'has_multiple_regions': False,
                    'region_count': 1,
                    'total_clusters': 1
                }
        
        except Exception as e:
            logger.warning(f"Color region detection error: {e}")
            return {
                'has_multiple_regions': False,
                'region_count': 1,
                'total_clusters': 1
            }
    
    def _detect_borders(self, gray, height, width):
        """Detect uniform borders that might indicate frames"""
        border_thickness = min(20, height//20, width//20)
        
        # Extract borders
        top_border = gray[:border_thickness, :]
        bottom_border = gray[-border_thickness:, :]
        left_border = gray[:, :border_thickness]
        right_border = gray[:, -border_thickness:]
        
        # Calculate variance in borders
        top_var = np.var(top_border)
        bottom_var = np.var(bottom_border)
        left_var = np.var(left_border)
        right_var = np.var(right_border)
        
        # Check for uniform borders (low variance)
        has_uniform_border = bool(top_var < 500 or bottom_var < 500 or 
                             left_var < 500 or right_var < 500)
        
        return {
            'has_uniform_border': has_uniform_border,
            'top_variance': round(float(top_var), 2),
            'bottom_variance': round(float(bottom_var), 2),
            'left_variance': round(float(left_var), 2),
            'right_variance': round(float(right_var), 2)
        }
    
    def _analyze_texture(self, gray):
        """Analyze image texture complexity"""
        # Calculate gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Calculate texture statistics
        mean_gradient = np.mean(gradient_magnitude)
        variance_gradient = np.var(gradient_magnitude)
        
        return {
            'mean': round(float(mean_gradient), 2),
            'variance': round(float(variance_gradient), 2)
        }
    
    def _calculate_collage_score(self, line_analysis, region_analysis, 
                                  border_analysis, texture_analysis):
        """Calculate overall collage probability score"""
        score = 0.0
        total_weights = 0
        
        # Line analysis weight (30%)
        if line_analysis['has_lines']:
            line_score = min(1.0, line_analysis['line_count'] / 10)
            score += 0.3 * line_score
        total_weights += 0.3
        
        # Region analysis weight (35%)
        if region_analysis['has_multiple_regions']:
            region_score = min(1.0, region_analysis['region_count'] / 5)
            score += 0.35 * region_score
        total_weights += 0.35
        
        # Border analysis weight (15%)
        if border_analysis['has_uniform_border']:
            score += 0.15
        total_weights += 0.15
        
        # Texture analysis weight (20%)
        texture_score = min(1.0, texture_analysis['variance'] / 2000)
        score += 0.20 * texture_score
        total_weights += 0.20
        
        # Normalize score
        return score / total_weights if total_weights > 0 else 0
    
    def _determine_specific_type(self, is_collage, line_analysis, region_analysis, border_analysis):
        """Determine specific image type"""
        if not is_collage:
            if border_analysis['has_uniform_border']:
                return "Single Image with Border/Frame"
            else:
                return "Full Single Image"
        else:
            if line_analysis['line_count'] > 10:
                return "Grid Collage (multiple images arranged in grid)"
            elif region_analysis['has_multiple_regions'] and region_analysis['region_count'] > 3:
                return "Photo Collage (overlapping or artistic arrangement)"
            else:
                return "Joined Image (multiple images combined)"
    
    def _get_recommendations(self, is_collage, specific_type):
        """Generate recommendations based on classification"""
        recommendations = []
        
        if is_collage:
            recommendations.append("This appears to be a collage or joined image")
            recommendations.append("Individual segments may need separate processing for best results")
            recommendations.append("Consider cropping individual images before applying filters")
            recommendations.append("For collages, try applying the same filter to see unified effect")
        else:
            recommendations.append("This is a single image - perfect for processing")
            recommendations.append("You can safely apply filters and denoising to the entire image")
            if specific_type == "Single Image with Border/Frame":
                recommendations.append("The border/frame will be preserved during processing")
            else:
                recommendations.append("Full image processing will work optimally")
        
        return recommendations
