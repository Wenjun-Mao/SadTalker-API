# SadTalker-API

**SadTalker** is a project that generates realistic 3D motion coefficients for stylized audio-driven single-image talking face animation. This repository contains the implementation of the SadTalker algorithm, offering both a WebUI (via Gradio) and a Command Line Interface (CLI) for generating talking head videos from a source image and an audio file.

## Project Structure

*   **`webui.bat`**: The primary entry point for Windows users. It automatically creates a Python virtual environment (`venv`), activates it, and runs `launcher.py`.
*   **`webui.sh`**: The entry point for Linux/macOS users.
*   **`launcher.py`**: A setup and launch script. It checks the Python version, installs necessary dependencies (including PyTorch and those in `requirements.txt`), and starts the Gradio WebUI (`app_sadtalker.py`).
*   **`app_sadtalker.py`**: The main application script defining the Gradio interface.
*   **`inference.py`**: The CLI entry point for generating videos without the WebUI.
*   **`requirements.txt`**: List of Python dependencies (used by `launcher.py`).
*   **`scripts/download_models.sh`**: Helper script to download the required pre-trained models.
*   **`src/`**: Contains the core source code for the models (`audio2exp_models`, `audio2pose_models`, `face3d`, `facerender`).

## Setup and Usage

### Prerequisites

*   **Python:** Version 3.10 is recommended (especially for Windows).
*   **Git:** Required for cloning and some setup operations.
*   **FFmpeg:** Required for video processing.

### Running the WebUI (Windows)

1.  Ensure you have Python 3.10 and git installed.
2.  Run **`webui.bat`**.
    *   This script will create a `venv` directory if it doesn't exist.
    *   It will install PyTorch and other dependencies automatically via `launcher.py`.
    *   It will launch the Gradio interface in your default browser.

### Running the CLI

To use the CLI, you should preferably activate the environment created by `webui.bat` or set up your own:

1.  **Activate Environment:**
    *   Windows: `venv\Scripts\activate`
2.  **Run Inference:**
    ```bash
    python inference.py --driven_audio <audio_path> --source_image <image_path> --enhancer gfpgan
    ```
    *   Check `python inference.py --help` for all available options (e.g., `--still`, `--preprocess full`).

### Models

The application requires several pre-trained models.
*   They are typically placed in `checkpoints/` and `gfpgan/weights/`.
*   You can run `bash scripts/download_models.sh` (if you have a bash environment) or download them manually as described in the `README.md`.

## Development Conventions

*   **Environment:** The project relies heavily on a specific environment setup (PyTorch versions, etc.), managed by `launcher.py`. When modifying code, ensure you are testing within this environment.
*   **Style:** Standard Python coding style.
*   **Frameworks:**
    *   **PyTorch**: Core ML framework.
    *   **Gradio**: Used for the Web UI.
    *   **GFPGAN**: Used for face enhancement.
*   **Architecture:** The core logic is split into several modules under `src/`, handling different aspects of the pipeline (Audio-to-Expression, Audio-to-Pose, Face Rendering).
