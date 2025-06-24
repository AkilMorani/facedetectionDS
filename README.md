# Face Detection Deep Learning Project

## Overview
This project implements a binary classification model for face detection using transfer learning with the **EfficientNetB0** architecture pre-trained on ImageNet. The model is trained and evaluated on a dataset of face and non-face images, with comprehensive data preprocessing, augmentation, and empirical tuning to optimize performance. The project is implemented in a Google Colab notebook using TensorFlow, Keras, and OpenCV, with data stored on Google Drive.

### Objectives
- Preprocess and augment image data for robust model training.
- Utilize transfer learning with EfficientNetB0 for binary classification (face vs. non-face).
- Evaluate model performance using accuracy, confusion matrix, and AUC-ROC metrics.
- Perform empirical tuning to address overfitting and improve generalization.

## Prerequisites
- **Google Colab**: For running the notebook.
- **Google Drive**: To store and access the dataset.
- **Python Libraries**:
  - `tensorflow`, `keras` (for model building and training)
  - `opencv-python` (for image preprocessing)
  - `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` (for data handling and visualization)
  - `skimage` (for advanced image processing)

Install dependencies in Colab:
```bash
!pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn scikit-image
```

## Dataset
The dataset is stored in Google Drive under the path `/content/drive/MyDrive/Deep Learning Assign 3/facedetectionDS/`, with two subdirectories:
- `faces/`: Images containing faces.
- `notfaces/`: Images without faces.

Key dataset details:
- **Image Size**: Resized to 224x224 pixels.
- **Classes**: Binary (0: Not Face, 1: Face).
- **Split**:
  - Training: 5,632 images.
  - Validation: 1,406 images.
  - Test: 7,038 images.

## Project Structure
- **Data Preprocessing**:
  - Scaling and resizing images to 224x224.
  - Augmentation using `ImageDataGenerator` (rotation, shifts, shear, zoom, horizontal flip).
  - Advanced preprocessing: Grayscale conversion, thresholding, histogram equalization, Gaussian blur.
- **Model Architecture**:
  - Base model: EfficientNetB0 (pre-trained on ImageNet, `include_top=False`).
  - Additional layers: MaxPooling2D, Flatten, Dense (256 units, ReLU), Dropout (0.5), Dense (1 unit, sigmoid).
  - Optimizer: Adam with varying learning rates.
- **Training**:
  - Three rounds of empirical tuning with learning rates of 1e-3, 1e-4, and 1e-5.
  - Early stopping to prevent overfitting.
- **Evaluation**:
  - Metrics: Accuracy, precision, recall, F1-score, AUC-ROC.
  - Visualizations: Confusion matrix, ROC curve, training/validation accuracy plots.

## Setup Instructions
1. **Mount Google Drive**:
   - Run the following in Colab to access the dataset:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Ensure the dataset is available at `/content/drive/MyDrive/Deep Learning Assign 3/facedetectionDS/`.

2. **Install Dependencies**:
   - Install required libraries as listed above.

3. **Run the Notebook**:
   - Open `DL_Assignment_2.html` in a browser to view the code and outputs, or convert it to a `.ipynb` file using an HTML-to-IPython converter (e.g., `nbconvert`).
   - Execute cells sequentially in Colab to preprocess data, train the model, and evaluate results.

4. **File Structure**:
   ```
   /content/drive/MyDrive/Deep Learning Assign 3/
   ├── facedetectionDS/
   │   ├── faces/        # Face images
   │   └── notfaces/     # Non-face images
   └── DL_Assignment_2.html  # Project notebook
   ```

## Key Results
### Data Preprocessing
- **Scaling and Resizing**: All images resized to 224x224.
- **Augmentation**: Applied rotation, shifts, shear, zoom, and horizontal flip, with pixel values normalized to [0,1].
- **Advanced Preprocessing**: Grayscale conversion, binary thresholding, histogram equalization, and Gaussian blur for enhanced feature extraction.

### Model Performance
- **Initial Training (Round 1)**:
  - Test Accuracy: 93.73%
  - Test AUC: 0.9345
  - Confusion Matrix: [[1222, 92], [349, 5375]]
  - Observations: High training accuracy (~100%) but signs of overfitting (validation accuracy lower).

- **Fine-Tuning (Round 2)**:
  - Test Accuracy: 99.23%
  - Test AUC: 0.9797
  - Confusion Matrix: [[1261, 53], [1, 5723]]
  - Observations: Improved performance, reduced overfitting, better balance between classes.

- **Further Fine-Tuning (Round 3)**:
  - Test Accuracy: 100%
  - Test AUC: 1.0
  - Confusion Matrix: [[1314, 0], [0, 5724]]
  - Observations: Perfect classification, no errors, stable validation performance.

### Analysis
- **Overfitting**: Addressed through early stopping and lower learning rates in Rounds 2 and 3.
- **Class Imbalance**: Improved in later rounds, with perfect performance on both classes in Round 3.
- **AUC-ROC**: High AUC scores indicate strong discriminative ability, reaching 1.0 in Round 3.

## Usage
To replicate the project:
1. Ensure the dataset is correctly structured in Google Drive.
2. Run the notebook cells in sequence to:
   - Preprocess and augment images.
   - Train the model through three tuning rounds.
   - Evaluate and visualize results.
3. Review outputs (accuracy, confusion matrix, ROC curves) to assess model performance.

## Limitations and Future Work
- **Class Imbalance**: Initial rounds showed bias toward the majority class (faces). Future work could explore techniques like weighted loss or oversampling.
- **Dataset Size**: Limited to 7,038 images. Expanding the dataset could improve robustness.
- **Preprocessing**: Advanced preprocessing (e.g., Gaussian blur) may not always enhance performance; further experimentation is needed.
- **Production Deployment**: Integrate with a cloud backend (e.g., AWS/GCP) for real-time face detection.

## License
This project is for educational purposes and licensed under the MIT License.
