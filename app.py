from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import custom Python modules
from image_processor import ImageProcessor
from noise_analyzer import NoiseAnalyzer
from image_classifier import ImageClassifier

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'}
app.config['SECRET_KEY'] = os.urandom(24)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Python processors
image_processor = ImageProcessor()
noise_analyzer = NoiseAnalyzer()
image_classifier = ImageClassifier()

class ImageSession:
    """Session manager for image processing"""
    
    def __init__(self):
        self.sessions = {}
        self.max_sessions = 100
        self.session_timeout = 3600  # 1 hour
    
    def create_session(self, image_path, original_image, filename):
        """Create a new session"""
        session_id = str(uuid.uuid4())[:8]
        
        # Clean old sessions if needed
        if len(self.sessions) >= self.max_sessions:
            self._clean_old_sessions()
        
        self.sessions[session_id] = {
            'original_path': image_path,
            'original_image': original_image.copy(),
            'processed_image': original_image.copy(),
            'filename': filename,
            'created_at': datetime.now(),
            'last_access': datetime.now()
        }
        
        logger.info(f"Session created: {session_id}")
        return session_id
    
    def get_session(self, session_id):
        """Retrieve session data"""
        session = self.sessions.get(session_id)
        if session:
            session['last_access'] = datetime.now()
        return session
    
    def update_processed(self, session_id, processed_image):
        """Update processed image in session"""
        if session_id in self.sessions:
            self.sessions[session_id]['processed_image'] = processed_image.copy()
            self.sessions[session_id]['last_access'] = datetime.now()
            return True
        return False
    
    def delete_session(self, session_id):
        """Delete a session"""
        if session_id in self.sessions:
            # Delete file
            session = self.sessions[session_id]
            if os.path.exists(session['original_path']):
                try:
                    os.remove(session['original_path'])
                except:
                    pass
            del self.sessions[session_id]
            logger.info(f"Session deleted: {session_id}")
            return True
        return False
    
    def _clean_old_sessions(self):
        """Remove old sessions"""
        now = datetime.now()
        to_delete = []
        
        for sid, session in self.sessions.items():
            age = (now - session['last_access']).total_seconds()
            if age > self.session_timeout:
                to_delete.append(sid)
        
        for sid in to_delete:
            self.delete_session(sid)
        
        if to_delete:
            logger.info(f"Cleaned {len(to_delete)} old sessions")

# Global session manager
session_manager = ImageSession()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def image_to_base64(image):
    """Convert numpy image to base64 string"""
    if image is None:
        return None
    
    try:
        # Convert BGR to RGB for display
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL Image
        pil_img = Image.fromarray(image_rgb)
        
        # Save to bytes
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Image to base64 error: {e}")
        return None

def get_image_info(image):
    """Get image metadata"""
    if image is None:
        return {}
    
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    file_size = image.nbytes / (1024 * 1024)  # Size in MB
    
    return {
        'width': width,
        'height': height,
        'channels': channels,
        'size_mb': round(file_size, 2),
        'aspect_ratio': round(width / height, 2)
    }

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Image Regeneration Studio',
        'version': '2.0.0',
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(session_manager.sessions)
    })

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Handle image upload and analysis"""
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'File type not allowed. Allowed: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
            }), 400
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if original_image is None:
            return jsonify({'error': 'Could not read image file'}), 400
        
        # Save original
        original_filename = secure_filename(f"original_{uuid.uuid4().hex}.png")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        cv2.imwrite(filepath, original_image)
        
        # Create session
        session_id = session_manager.create_session(filepath, original_image, file.filename)
        
        # Get image info
        image_info = get_image_info(original_image)
        
        # Perform analysis
        logger.info(f"Analyzing image for session {session_id}")
        image_type_analysis = image_classifier.analyze(original_image)
        noise_analysis = noise_analyzer.analyze(original_image)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'original_image': image_to_base64(original_image),
            'image_info': image_info,
            'analysis': {
                'image_type': image_type_analysis,
                'noise': noise_analysis
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Re-analyze current image"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found or expired'}), 404
        
        original_image = session['original_image']
        
        # Perform analysis
        image_type_analysis = image_classifier.analyze(original_image)
        noise_analysis = noise_analyzer.analyze(original_image)
        
        return jsonify({
            'success': True,
            'image_type': image_type_analysis,
            'noise': noise_analysis
        }), 200
    
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500

@app.route('/api/denoise', methods=['POST'])
def denoise_image():
    """Apply intelligent denoising"""
    try:
        data = request.json
        session_id = data.get('session_id')
        strength = data.get('strength', 'medium')
        noise_type = data.get('noise_type', 'auto')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found or expired'}), 404
        
        original_image = session['original_image']
        
        # Auto-detect noise if requested
        if noise_type == 'auto':
            noise_analysis = noise_analyzer.analyze(original_image)
            noise_type = noise_analysis['type']
            logger.info(f"Auto-detected noise type: {noise_type}")
        
        # Apply denoising
        processed_image = image_processor.denoise(
            original_image,
            method='smart',
            strength=strength,
            noise_type=noise_type
        )
        
        # Update session
        session_manager.update_processed(session_id, processed_image)
        
        # Analyze result
        remaining_noise = noise_analyzer.analyze(processed_image)
        
        # Calculate improvement
        original_noise_level = noise_analyzer._estimate_noise_level(
            cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) if len(original_image.shape) == 3 else original_image
        )
        new_noise_level = remaining_noise['noise_level']
        
        improvement = max(0, original_noise_level['percentage'] - new_noise_level['percentage'])
        
        return jsonify({
            'success': True,
            'processed_image': image_to_base64(processed_image),
            'noise_analysis': remaining_noise,
            'denoise_method': noise_type,
            'strength_used': strength,
            'improvement_percentage': round(improvement, 1)
        }), 200
    
    except Exception as e:
        logger.error(f"Denoise error: {str(e)}")
        return jsonify({'error': f'Denoise error: {str(e)}'}), 500

@app.route('/api/filter', methods=['POST'])
def apply_filter():
    """Apply artistic filter"""
    try:
        data = request.json
        session_id = data.get('session_id')
        filter_type = data.get('filter_type', 'sharpen')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found or expired'}), 404
        
        original_image = session['original_image']
        
        # Apply filter
        processed_image = image_processor.apply_filter(original_image, filter_type)
        
        # Update session
        session_manager.update_processed(session_id, processed_image)
        
        return jsonify({
            'success': True,
            'processed_image': image_to_base64(processed_image),
            'filter_applied': filter_type
        }), 200
    
    except Exception as e:
        logger.error(f"Filter error: {str(e)}")
        return jsonify({'error': f'Filter error: {str(e)}'}), 500

@app.route('/api/regenerate', methods=['POST'])
def regenerate_image():
    """Regenerate/restore image"""
    try:
        data = request.json
        session_id = data.get('session_id')
        method = data.get('method', 'auto')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found or expired'}), 404
        
        original_image = session['original_image']
        
        # Regenerate image
        processed_image = image_processor.regenerate(original_image, method)
        
        # Update session
        session_manager.update_processed(session_id, processed_image)
        
        return jsonify({
            'success': True,
            'processed_image': image_to_base64(processed_image),
            'regeneration_method': method
        }), 200
    
    except Exception as e:
        logger.error(f"Regeneration error: {str(e)}")
        return jsonify({'error': f'Regeneration error: {str(e)}'}), 500

@app.route('/api/add-noise', methods=['POST'])
def add_artificial_noise():
    """Add artificial noise for testing"""
    try:
        data = request.json
        session_id = data.get('session_id')
        noise_type = data.get('noise_type', 'gaussian')
        intensity = int(data.get('intensity', 25))
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found or expired'}), 404
        
        original_image = session['original_image']
        
        # Add noise
        noisy_image = image_processor.add_noise(original_image, noise_type, intensity)
        
        # Update session
        session_manager.update_processed(session_id, noisy_image)
        
        # Analyze added noise
        noise_analysis = noise_analyzer.analyze(noisy_image)
        
        return jsonify({
            'success': True,
            'processed_image': image_to_base64(noisy_image),
            'noise_added': noise_type,
            'intensity': intensity,
            'noise_analysis': noise_analysis
        }), 200
    
    except Exception as e:
        logger.error(f"Add noise error: {str(e)}")
        return jsonify({'error': f'Add noise error: {str(e)}'}), 500

@app.route('/api/reset', methods=['POST'])
def reset_image():
    """Reset to original image"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found or expired'}), 404
        
        # Reset to original
        session_manager.update_processed(session_id, session['original_image'])
        
        return jsonify({
            'success': True,
            'message': 'Reset to original image'
        }), 200
    
    except Exception as e:
        logger.error(f"Reset error: {str(e)}")
        return jsonify({'error': f'Reset error: {str(e)}'}), 500

@app.route('/api/download', methods=['POST'])
def download_image():
    """Download processed image"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found or expired'}), 404
        
        processed_image = session['processed_image']
        
        # Convert to bytes
        _, buffer = cv2.imencode('.png', processed_image)
        io_buf = BytesIO(buffer)
        
        # Generate filename
        original_name = session['filename'].rsplit('.', 1)[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{original_name}_processed_{timestamp}.png"
        
        return send_file(
            io_buf,
            mimetype='image/png',
            as_attachment=True,
            download_name=filename
        ), 200
    
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': f'Download error: {str(e)}'}), 500

@app.route('/api/batch-process', methods=['POST'])
def batch_process():
    """Process multiple images"""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No image files provided'}), 400
        
        files = request.files.getlist('images')
        operation = request.form.get('operation', 'denoise')
        strength = request.form.get('strength', 'medium')
        
        if len(files) > 10:
            return jsonify({'error': 'Maximum 10 images per batch'}), 400
        
        results = []
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    file_bytes = np.frombuffer(file.read(), np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        if operation == 'denoise':
                            processed = image_processor.denoise(image, method='smart', strength=strength)
                        elif operation == 'sharpen':
                            processed = image_processor.apply_filter(image, 'sharpen')
                        elif operation == 'cartoon':
                            processed = image_processor.apply_filter(image, 'cartoon')
                        else:
                            processed = image
                        
                        results.append({
                            'filename': file.filename,
                            'success': True,
                            'image': image_to_base64(processed)
                        })
                    else:
                        results.append({
                            'filename': file.filename,
                            'success': False,
                            'error': 'Could not read image'
                        })
                except Exception as e:
                    results.append({
                        'filename': file.filename,
                        'success': False,
                        'error': str(e)
                    })
        
        return jsonify({
            'success': True,
            'processed_count': len([r for r in results if r['success']]),
            'total_count': len(files),
            'results': results
        }), 200
    
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        return jsonify({'error': f'Batch processing error: {str(e)}'}), 500

@app.route('/api/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session"""
    try:
        if session_manager.delete_session(session_id):
            return jsonify({'success': True, 'message': 'Session deleted'}), 200
        else:
            return jsonify({'error': 'Session not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("AI Image Analysis & Regeneration Studio")
    print("=" * 60)
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Server: http://localhost:5000")
    print(f"API docs: http://localhost:5000/health")
    print("=" * 60)
    print("Press CTRL+C to stop the server")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
