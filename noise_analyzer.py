import cv2
import numpy as np
from scipy import stats
from scipy.fft import fft2, fftshift
import logging

logger = logging.getLogger(__name__)

class NoiseAnalyzer:
    """Noise detection and analysis module"""
    
    def __init__(self):
        self.noise_types = ['Gaussian', 'Salt & Pepper', 'Poisson', 'Speckle', 'Periodic']
    
    def analyze(self, image):
        """
        Analyze image and detect noise type
        
        Args:
            image: numpy array image (BGR format)
        
        Returns:
            Dictionary with noise analysis results
        """
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Calculate basic statistics
            stats_dict = self._calculate_statistics(gray)
            
            # Calculate noise scores
            noise_scores = self._calculate_noise_scores(gray, stats_dict)
            
            # Determine primary noise type
            detected_noise = max(noise_scores, key=noise_scores.get)
            confidence = noise_scores[detected_noise]
            
            # Estimate noise level
            noise_level = self._estimate_noise_level(gray)
            
            return {
                'type': detected_noise,
                'confidence': round(confidence * 100, 1),
                'noise_level': noise_level,
                'scores': {k: round(v * 100, 1) for k, v in noise_scores.items()},
                'statistics': stats_dict,
                'recommendations': self._get_recommendations(detected_noise, confidence, noise_level)
            }
        
        except Exception as e:
            logger.error(f"Noise analysis error: {e}")
            return {
                'type': 'Unknown',
                'confidence': 0,
                'noise_level': {'level': 'Unknown', 'percentage': 0},
                'scores': {},
                'statistics': {},
                'recommendations': ['Unable to analyze noise. Please try another image.']
            }
    
    def _calculate_statistics(self, gray):
        """Calculate statistical properties of the image"""
        flattened = gray.flatten()
        
        # Basic statistics
        mean_val = np.mean(flattened)
        variance = np.var(flattened)
        std_dev = np.std(flattened)
        
        # Advanced statistics
        skewness = stats.skew(flattened)
        kurtosis = stats.kurtosis(flattened)
        
        # Histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Edge statistics
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Frequency domain analysis
        f_transform = fft2(gray)
        f_shift = fftshift(f_transform)
        magnitude = np.abs(f_shift)
        magnitude_flat = magnitude.flatten()
        freq_mean = np.mean(magnitude_flat)
        freq_std = np.std(magnitude_flat)
        
        # Local variance analysis
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_sq_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        local_var = local_sq_mean - local_mean**2
        mean_local_var = np.mean(local_var)
        
        return {
            'mean': round(float(mean_val), 2),
            'variance': round(float(variance), 2),
            'std_dev': round(float(std_dev), 2),
            'skewness': round(float(skewness), 3),
            'kurtosis': round(float(kurtosis), 3),
            'entropy': round(float(entropy), 3),
            'edge_density': round(float(edge_density), 3),
            'freq_mean': round(float(freq_mean), 2),
            'freq_std': round(float(freq_std), 2),
            'mean_local_variance': round(float(mean_local_var), 2)
        }
    
    def _calculate_noise_scores(self, gray, stats):
        """Calculate probability scores for each noise type"""
        scores = {}
        
        # Gaussian noise detection
        # Gaussian noise typically has low skewness, kurtosis near 3, and high variance
        if abs(stats['skewness']) < 0.5 and abs(stats['kurtosis']) < 1:
            gaussian_score = min(1.0, stats['variance'] / 500)
        else:
            gaussian_score = max(0, 1 - abs(stats['skewness']) / 2)
        scores['Gaussian'] = gaussian_score
        
        # Salt & Pepper noise detection
        # Look for extreme pixel values (0 and 255)
        salt_pepper_ratio = (np.sum(gray == 0) + np.sum(gray == 255)) / gray.size
        if salt_pepper_ratio > 0.05:
            sp_score = min(1.0, salt_pepper_ratio * 10)
        else:
            sp_score = salt_pepper_ratio * 10
        scores['Salt & Pepper'] = sp_score
        
        # Poisson noise detection
        # Poisson noise has mean ≈ variance
        mean_var_ratio = stats['variance'] / stats['mean'] if stats['mean'] > 0 else 0
        if 0.8 < mean_var_ratio < 1.2:
            poisson_score = min(1.0, mean_var_ratio)
        else:
            poisson_score = max(0, 1 - abs(mean_var_ratio - 1))
        scores['Poisson'] = poisson_score
        
        # Speckle noise detection
        # Speckle noise has high local variance relative to mean
        speckle_indicator = stats['mean_local_variance'] / (stats['mean']**2) if stats['mean'] > 0 else 0
        if speckle_indicator > 0.1:
            speckle_score = min(1.0, speckle_indicator)
        else:
            speckle_score = speckle_indicator * 5
        scores['Speckle'] = speckle_score
        
        # Periodic noise detection
        # Periodic noise shows peaks in frequency domain
        freq_ratio = stats['freq_std'] / stats['freq_mean'] if stats['freq_mean'] > 0 else 0
        if freq_ratio > 2:
            periodic_score = min(1.0, freq_ratio / 5)
        else:
            periodic_score = freq_ratio / 2
        scores['Periodic'] = periodic_score
        
        return scores
    
    def _estimate_noise_level(self, gray):
        """Estimate the overall noise level in the image"""
        # Calculate noise using median absolute deviation
        median = np.median(gray)
        mad = np.median(np.abs(gray - median))
        noise_estimate = mad / 0.6745  # Convert to standard deviation estimate
        
        # Normalize to percentage
        noise_percentage = min(100, (noise_estimate / 255) * 100)
        
        if noise_percentage < 5:
            level = 'Very Low'
        elif noise_percentage < 15:
            level = 'Low'
        elif noise_percentage < 30:
            level = 'Medium'
        elif noise_percentage < 50:
            level = 'High'
        else:
            level = 'Very High'
        
        return {
            'percentage': round(noise_percentage, 1),
            'level': level,
            'estimated_std': round(float(noise_estimate), 2)
        }
    
    def _get_recommendations(self, noise_type, confidence, noise_level):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if noise_type == 'Gaussian':
            recommendations.append("Use Gaussian blur or Non-Local Means denoising")
            if confidence > 70:
                recommendations.append("Medium to strong denoising recommended")
            else:
                recommendations.append("Try weak to medium denoising first")
        
        elif noise_type == 'Salt & Pepper':
            recommendations.append("Use Median filter for best results")
            recommendations.append("This noise type responds very well to median filtering")
        
        elif noise_type == 'Poisson':
            recommendations.append("Use Non-Local Means denoising")
            recommendations.append("Poisson noise is intensity-dependent")
        
        elif noise_type == 'Speckle':
            recommendations.append("Use Bilateral filter or Wavelet denoising")
            recommendations.append("Speckle noise requires edge-preserving methods")
        
        elif noise_type == 'Periodic':
            recommendations.append("Use Fourier transform filtering")
            recommendations.append("Try median filter in frequency domain")
        
        else:
            recommendations.append("Try Smart Denoise for automatic noise removal")
        
        # Add strength recommendation based on noise level
        if noise_level['level'] in ['High', 'Very High']:
            recommendations.append("Use strong denoising for best results")
        elif noise_level['level'] == 'Medium':
            recommendations.append("Medium denoising strength should work well")
        else:
            recommendations.append("Weak denoising is sufficient")
        
        return recommendations
