Deepfake Detector AI

This project is a full-stack application designed to detect deepfakes in both images and videos using advanced deep learning models. It features a React-based web interface for user-friendly analysis and a powerful Python backend powered by PyTorch.

<!-- Optional: Add a screenshot of your UI -->

Features

Dual-Mode Analysis: Classify both individual/multiple images and videos.

Advanced AI Models:

Image Analysis: Utilizes a fine-tuned SMDNet model for high-accuracy spatial artifact detection.

Video Analysis: Employs a hybrid CNN+RNN (LSTM) architecture to analyze temporal inconsistencies across video frames.

Web-Based UI: A clean, modern, and responsive user interface built with React and Tailwind CSS.

REST API Backend: A Flask-based server that exposes the PyTorch models for inference.

Tech Stack

Backend: Python, PyTorch, Flask, OpenCV

Frontend: React, TypeScript, Vite, Tailwind CSS

Models: Custom CNNs (SMDNet), CNN+RNN (EfficientNet+LSTM)

Setup and Installation

Prerequisites

Python 3.10+

Node.js and npm

Git and Git LFS

1. Clone the Repository

git clone [Your-Repo-URL-Here]
cd DeepFake


2. Backend Setup

# Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt # We will create this file later


3. Frontend Setup

# Navigate to the website directory
cd website

# Install Node.js dependencies
npm install


How to Run

You need to run the backend and frontend servers in two separate terminals.

1. Start the Backend Server:

Terminal 1: Navigate to DeepFake/website/backend/

Activate the virtual environment: ../../venv/Scripts/activate

Run the server: python app.py

The backend will be running on http://localhost:5000.

2. Start the Frontend Server:

Terminal 2: Navigate to DeepFake/website/

Run the server: npm run dev

The UI will be accessible at http://localhost:5173 (or a similar address).


---

### Step 3: Handling the Large `.pth` Model Files with Git LFS

Your `.pth` model files are too large for standard Git. We must use **Git Large File Storage (LFS)**. It's an extension for Git that stores large files on a separate server, keeping your main repository small and fast.

**Action:**
1.  **Install Git LFS:** If you don't have it, download and install it from the [official Git LFS website](https://git-lfs.github.com/).
2.  **Set up LFS in your project:** Open your terminal in the `DeepFake/` root directory and run this command **only once**:
    ```bash
    git lfs install
    ```
3.  **Tell LFS which files to track:** Now, tell LFS to handle all `.pth` files. Run this command:
    ```bash
    git lfs track "*.pth"
