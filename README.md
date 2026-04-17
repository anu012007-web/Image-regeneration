# AI Image Analysis & Regeneration Studio

A powerful web-based application built with **Flask** and **OpenCV** designed to analyze, denoise, and enhance images using intelligent algorithms.

## Features

- **Image Analysis**: Detects image types and analyzes noise profiles automatically.
- **Intelligent Denoising**: Removes noise from images using smart techniques tailored to the specific noise profile.
- **Artificial Noise Addition**: Simulate different types of noise (e.g., Gaussian) for testing robust regeneration.
- **Artistic Filters**: Apply sharpening or cartoon-style filters to enhance image aesthetics.
- **Batch Processing**: Process multiple images at once (up to 10 per batch).
- **Session Management**: Uploads are managed in a session-based environment. Auto-cleaning functionality handles expired sessions efficiently.
- **RESTful API**: Supports robust API endpoints built with Flask to handle image operations effortlessly.

## Technologies Used

- **Backend**: Python, Flask, Flask-CORS, Werkzeug
- **Image Processing**: OpenCV (opencv-python), Pillow (PIL), NumPy, SciPy
- **Machine Learning / Algorithmic Analysis**: Scikit-Learn, PyWavelets
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla API Integration)

## Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/anu012007-web/Image-regeneration.git
   cd Image-regeneration
   ```

2. **Set up a Virtual Environment (Optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Linux/Mac
   # venv\Scripts\activate        # On Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python app.py
   ```

5. **Access the App**
   Open your browser and navigate to:
   [http://localhost:5000](http://localhost:5000)

## API Endpoints Overview

- `GET /health`: Check service health and version.
- `POST /api/upload`: Upload an image and generate an analysis profile.
- `POST /api/analyze`: Re-analyze the uploaded image.
- `POST /api/denoise`: Apply intelligent denoising.
- `POST /api/filter`: Apply artistic filters like 'sharpen' or 'cartoon'.
- `POST /api/regenerate`: Restore/regenerate image conditionally.
- `POST /api/add-noise`: Inject artificial noise.
- `POST /api/download`: Download the processed image.
- `POST /api/batch-process`: Queue multiple files for batched manipulation.

## Project Structure

- `app.py`: Main Flask application handling routes, session management, and APIs.
- `image_processor.py`: Core processing logic utilizing OpenCV to manipulate, enrich, and filter images.
- `noise_analyzer.py`: Responsible for estimating noise levels and determining noise types.
- `image_classifier.py`: Analyzes basic image features and classification behaviors.
- `templates/`: HTML templates (e.g., `index.html`) for the frontend interface.
- `static/`: Contains user-uploaded content and stylesheets structure (`style.css`).

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you might want to change.
