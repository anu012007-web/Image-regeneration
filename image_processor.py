import cv2
import numpy as np
from scipy import ndimage
import pywt
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Core image processing operations"""
    
    def __init__(self):
        self.supported_filters = [
            'sharpen', 'blur', 'gaussian_blur', 'median_blur',
            'edge_detection', 'emboss', 'cartoon', 'sketch',
            'oil_paint', 'watercolor', 'bilateral_filter'
        ]
        
        self.supported_noise_types = ['gaussian', 'salt_pepper', 'poisson', 'speckle', 'periodic']
        
        # Filter kernels
        self.kernels = {
            'sharpen': np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
            'emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
            'edge_detect': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
            'blur': np.ones((5, 5), np.float32) / 25
        }
    
    def denoise(self, image, method='smart', strength='medium', noise_type=None):
        """
        Remove noise from image
        
        Args:
            image: numpy array image (BGR format)
            method: 'smart', 'gaussian', 'median', 'bilateral', 'nl_means', 'wavelet'
            strength: 'weak', 'medium', 'strong'
            noise_type: type of noise for smart denoising
        
        Returns:
            Denoised image
        """
        # Parameter mapping
        strength_params = {
            'weak': {'h': 5, 'kernel_size': 3, 'd': 5, 'sigma_color': 50, 'sigma_space': 50},
            'medium': {'h': 10, 'kernel_size': 5, 'd': 9, 'sigma_color': 75, 'sigma_space': 75},
            'strong': {'h': 15, 'kernel_size': 7, 'd': 15, 'sigma_color': 100, 'sigma_space': 100}
        }
        
        params = strength_params.get(strength, strength_params['medium'])
        
        try:
            if method == 'smart':
                if noise_type == 'Gaussian':
                    return self._gaussian_denoise(image, params['kernel_size'])
                elif noise_type == 'Salt & Pepper':
                    return self._median_denoise(image, params['kernel_size'])
                elif noise_type == 'Poisson':
                    return self._nl_means_denoise(image, params['h'])
                elif noise_type == 'Speckle':
                    return self._bilateral_denoise(image, params['d'], params['sigma_color'], params['sigma_space'])
                else:
                    return self._nl_means_denoise(image, params['h'])
            
            elif method == 'gaussian':
                return self._gaussian_denoise(image, params['kernel_size'])
            
            elif method == 'median':
                return self._median_denoise(image, params['kernel_size'])
            
            elif method == 'bilateral':
                return self._bilateral_denoise(image, params['d'], params['sigma_color'], params['sigma_space'])
            
            elif method == 'nl_means':
                return self._nl_means_denoise(image, params['h'])
            
            elif method == 'wavelet':
                return self._wavelet_denoise(image)
            
            else:
                logger.warning(f"Unknown denoise method: {method}, using original")
                return image
        
        except Exception as e:
            logger.error(f"Denoise error: {e}")
            return image
    
    def _gaussian_denoise(self, image, kernel_size):
        """Apply Gaussian blur"""
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        if len(image.shape) == 3:
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def _median_denoise(self, image, kernel_size):
        """Apply median filter"""
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        if len(image.shape) == 3:
            return cv2.medianBlur(image, kernel_size)
        return cv2.medianBlur(image, kernel_size)
    
    def _bilateral_denoise(self, image, d, sigma_color, sigma_space):
        """Apply bilateral filter"""
        if len(image.shape) == 3:
            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def _nl_means_denoise(self, image, h):
        """Apply Non-Local Means denoising"""
        template_window = 7
        search_window = 21
        
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, h, h, template_window, search_window)
        return cv2.fastNlMeansDenoising(image, None, h, template_window, search_window)
    
    def _wavelet_denoise(self, image):
        """Wavelet-based denoising"""
        try:
            if len(image.shape) == 3:
                denoised_channels = []
                for i in range(3):
                    channel = image[:, :, i].astype(np.float32)
                    coeffs = pywt.wavedec2(channel, 'db4', level=2)
                    
                    # Estimate noise standard deviation
                    sigma = np.median(np.abs(coeffs[-1][0])) / 0.6745
                    threshold = sigma * np.sqrt(2 * np.log(channel.size))
                    
                    # Apply soft thresholding
                    coeffs_thresh = list(coeffs)
                    coeffs_thresh[1:] = [
                        (pywt.threshold(c, threshold, mode='soft'),
                         pywt.threshold(d, threshold, mode='soft'))
                        for c, d in coeffs[1:]
                    ]
                    
                    denoised = pywt.waverec2(coeffs_thresh, 'db4')
                    denoised_channels.append(denoised[:channel.shape[0], :channel.shape[1]])
                
                return np.stack(denoised_channels, axis=2).astype(np.uint8)
            else:
                channel = image.astype(np.float32)
                coeffs = pywt.wavedec2(channel, 'db4', level=2)
                sigma = np.median(np.abs(coeffs[-1][0])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(channel.size))
                coeffs_thresh = list(coeffs)
                coeffs_thresh[1:] = [
                    (pywt.threshold(c, threshold, mode='soft'),
                     pywt.threshold(d, threshold, mode='soft'))
                    for c, d in coeffs[1:]
                ]
                denoised = pywt.waverec2(coeffs_thresh, 'db4')
                return denoised[:image.shape[0], :image.shape[1]].astype(np.uint8)
        except Exception as e:
            logger.warning(f"Wavelet denoise failed, falling back to bilateral: {e}")
            return self._bilateral_denoise(image, 9, 75, 75)
    
    def apply_filter(self, image, filter_type):
        """Apply artistic filter to image"""
        try:
            filter_type = filter_type.lower()
            
            if filter_type == 'sharpen':
                return cv2.filter2D(image, -1, self.kernels['sharpen'])
            
            elif filter_type == 'blur':
                return cv2.filter2D(image, -1, self.kernels['blur'])
            
            elif filter_type == 'gaussian_blur':
                return cv2.GaussianBlur(image, (5, 5), 0)
            
            elif filter_type == 'median_blur':
                return cv2.medianBlur(image, 5)
            
            elif filter_type == 'edge_detection':
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            elif filter_type == 'emboss':
                return cv2.filter2D(image, -1, self.kernels['emboss'])
            
            elif filter_type == 'cartoon':
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)
                edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, 9, 9)
                color = cv2.bilateralFilter(image, 9, 300, 300)
                return cv2.bitwise_and(color, color, mask=edges)
            
            elif filter_type == 'sketch':
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                inv = 255 - gray
                blur = cv2.GaussianBlur(inv, (21, 21), 0)
                inv_blur = 255 - blur
                sketch = cv2.divide(gray, inv_blur, scale=256.0)
                return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
            
            elif filter_type == 'oil_paint':
                return cv2.medianBlur(image, 7)
            
            elif filter_type == 'watercolor':
                return cv2.stylization(image, sigma_s=60, sigma_r=0.6)
            
            elif filter_type == 'bilateral_filter':
                return cv2.bilateralFilter(image, 9, 75, 75)
            
            else:
                logger.warning(f"Unknown filter type: {filter_type}")
                return image
        
        except Exception as e:
            logger.error(f"Filter application error: {e}")
            return image
    
    def regenerate(self, image, method='auto'):
        """
        Regenerate/restore image using various methods
        
        Args:
            image: numpy array image
            method: 'auto', 'median', 'gaussian', 'bilateral', 'nl_means', 'wavelet'
        
        Returns:
            Regenerated image
        """
        try:
            if method == 'auto':
                # Auto-select best method based on image characteristics
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                variance = np.var(gray)
                
                if variance > 1000:
                    # High variance - use median filter
                    return self.denoise(image, method='median', strength='medium')
                elif variance > 500:
                    # Medium variance - use bilateral filter
                    return self.denoise(image, method='bilateral', strength='medium')
                else:
                    # Low variance - use wavelet denoising
                    return self.denoise(image, method='wavelet', strength='weak')
            
            elif method == 'median':
                return self.denoise(image, method='median', strength='medium')
            
            elif method == 'gaussian':
                return self.denoise(image, method='gaussian', strength='weak')
            
            elif method == 'bilateral':
                return self.denoise(image, method='bilateral', strength='medium')
            
            elif method == 'nl_means':
                return self.denoise(image, method='nl_means', strength='medium')
            
            elif method == 'wavelet':
                return self._wavelet_denoise(image)
            
            else:
                logger.warning(f"Unknown regeneration method: {method}")
                return image
        
        except Exception as e:
            logger.error(f"Regeneration error: {e}")
            return image
    
    def add_noise(self, image, noise_type='gaussian', intensity=25):
        """
        Add artificial noise to image for testing
        
        Args:
            image: numpy array image
            noise_type: 'gaussian', 'salt_pepper', 'poisson', 'speckle', 'periodic'
            intensity: noise intensity (0-100)
        
        Returns:
            Noisy image
        """
        try:
            noisy = image.copy().astype(np.float32)
            
            if noise_type == 'gaussian':
                # Add Gaussian noise
                noise = np.random.normal(0, intensity, image.shape)
                noisy = image + noise
            
            elif noise_type == 'salt_pepper':
                # Add Salt & Pepper noise
                salt_vs_pepper = 0.5
                amount = intensity / 1000
                num_salt = np.ceil(amount * image.size * salt_vs_pepper)
                num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))
                
                # Add salt (white pixels)
                coords = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape]
                noisy[tuple(coords)] = 255
                
                # Add pepper (black pixels)
                coords = [np.random.randint(0, i-1, int(num_pepper)) for i in image.shape]
                noisy[tuple(coords)] = 0
            
            elif noise_type == 'poisson':
                # Add Poisson noise
                # Scale image to avoid negative values
                scaled = np.maximum(image, 0)
                noisy = np.random.poisson(scaled).astype(np.float32)
            
            elif noise_type == 'speckle':
                # Add Speckle noise (multiplicative)
                noise = np.random.normal(0, intensity/100, image.shape)
                noisy = image + image * noise
            
            elif noise_type == 'periodic':
                # Add periodic/sinusoidal noise
                rows, cols = image.shape[:2]
                x = np.arange(cols)
                y = np.arange(rows)
                X, Y = np.meshgrid(x, y)
                period = max(10, intensity)
                amplitude = 50
                noise = amplitude * np.sin(2 * np.pi * X / period) * np.sin(2 * np.pi * Y / period)
                if len(image.shape) == 3:
                    noise = np.stack([noise] * 3, axis=2)
                noisy = image + noise
            
            else:
                logger.warning(f"Unknown noise type: {noise_type}")
                return image
            
            # Clip values to valid range
            return np.clip(noisy, 0, 255).astype(np.uint8)
        
        except Exception as e:
            logger.error(f"Add noise error: {e}")
            return image
    
    def enhance_contrast(self, image, method='histogram'):
        """
        Enhance image contrast
        
        Args:
            image: numpy array image
            method: 'histogram', 'clahe', 'gamma'
        
        Returns:
            Contrast enhanced image
        """
        try:
            if method == 'histogram':
                if len(image.shape) == 3:
                    # Convert to YUV, equalize Y channel
                    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                else:
                    return cv2.equalizeHist(image)
            
            elif method == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                if len(image.shape) == 3:
                    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
                    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                else:
                    return clahe.apply(image)
            
            elif method == 'gamma':
                # Gamma correction
                gamma = 1.5
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
                return cv2.LUT(image, table)
            
            else:
                return image
        
        except Exception as e:
            logger.error(f"Contrast enhancement error: {e}")
            return image
    
    def resize_image(self, image, width=None, height=None, scale=None):
        """
        Resize image maintaining aspect ratio
        
        Args:
            image: numpy array image
            width: target width
            height: target height
            scale: scale factor
        
        Returns:
            Resized image
        """
        try:
            h, w = image.shape[:2]
            
            if scale:
                new_w = int(w * scale)
                new_h = int(h * scale)
            elif width and height:
                new_w, new_h = width, height
            elif width:
                ratio = width / w
                new_w = width
                new_h = int(h * ratio)
            elif height:
                ratio = height / h
                new_h = height
                new_w = int(w * ratio)
            else:
                return image
            
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        except Exception as e:
            logger.error(f"Resize error: {e}")
            return image
    
    def rotate_image(self, image, angle):
        """Rotate image by specified angle"""
        try:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, matrix, (w, h))
        except Exception as e:
            logger.error(f"Rotate error: {e}")
            return image
    
    def adjust_brightness(self, image, value):
        """Adjust image brightness"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        except Exception as e:
            logger.error(f"Brightness adjustment error: {e}")
            return image
