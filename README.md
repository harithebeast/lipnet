# LipNet - Lip Reading with Deep Learning

This project implements LipNet, a deep learning model for lip reading using TensorFlow. The model can predict spoken words from video frames of lip movements.

## Features

- Real-time lip reading prediction
- Web interface using Streamlit
- Support for video processing and frame extraction
- Integration with the original LipNet model architecture

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/harithebeast/lipnet.git
cd lipnet
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the pre-trained model:
- Create a `models` directory in the project root
- Download the pre-trained model weights and place them in the `models` directory

5. Run the application:
```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main Streamlit application
- `modelutil.py`: Model loading and prediction utilities
- `utils.py`: Helper functions for data processing
- `LipNet.ipynb`: Jupyter notebook with model training code
- `data/`: Directory containing video data and alignments
- `models/`: Directory for storing model weights

## Usage

1. Launch the application using Streamlit
2. Select a video from the dropdown menu
3. View the original video and the model's prediction
4. Compare the prediction with the ground truth text

## Requirements

- Python 3.7+
- TensorFlow 2.8.0+
- OpenCV
- FFmpeg
- Other dependencies listed in requirements.txt

## License

This project is licensed under the MIT License - see the LICENSE file for details. 