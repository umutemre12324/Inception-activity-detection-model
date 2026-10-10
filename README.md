# Inception-Based Human Activity Recognition

A deep learning project for **Human Activity Recognition (HAR)** using the UCI Human Activity Recognition Using Smartphones dataset. The project uses an Inception-based neural network to classify human activities from smartphone sensor signals.

## Overview

Human Activity Recognition aims to identify physical activities using data collected from sensors such as accelerometers and gyroscopes.

In this project, an Inception-based deep learning model is designed to learn temporal patterns from sensor data and classify different human activities.

## Dataset

The project uses the [UCI Human Activity Recognition Using Smartphones Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

The dataset contains smartphone sensor measurements collected from 30 participants performing six activities:

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

The original dataset provides separate training and test sets.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

## Project Structure

```text
inception_activtiy_project/
├── dataset.py
├── prepocessing.py
├── model.py
├── train.py
├── test.py
├── .gitignore
└── README.md
```

Generated datasets, trained model files, and preprocessing artifacts are excluded from version control.

## Methodology

### 1. Data Loading

The raw sensor signals and activity labels are loaded from the UCI HAR dataset.

### 2. Data Preprocessing

- Sensor signals are loaded and organized into suitable arrays.
- Activity labels are converted to one-hot encoded vectors.
- `StandardScaler` is used to normalize sensor features.
- Preprocessed arrays and fitted scalers are saved for subsequent stages.

### 3. Model Architecture

An Inception-based neural network is implemented using TensorFlow and Keras.

The model is designed to extract useful features from sensor signals through convolutional operations and classify activities into the corresponding categories.

### 4. Model Training

The model is trained using categorical cross-entropy loss and the Adam optimizer.

Training includes:
- Model checkpointing
- Early stopping
- Validation accuracy and loss monitoring

### 5. Evaluation

The trained model can be evaluated using the test dataset. Accuracy, loss, and training curves can be used to analyze model performance and potential overfitting.

## Installation

Clone the repository:

```bash
git clone https://github.com/umutemre12324/Inception-activity-detection-model.git
cd Inception-activity-detection-model
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the required libraries:

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow numpy pandas scikit-learn matplotlib joblib
```

## Usage

Run the scripts in the following order:

**1. Prepare the dataset**

```bash
python dataset.py
```

**2. Preprocess the data**

```bash
python prepocessing.py
```

**3. Train the model**

```bash
python train.py
```

**4. Evaluate the model**

```bash
python test.py
```

Make sure the UCI HAR dataset is downloaded and extracted in the location expected by `dataset.py` before running the scripts.

## Future Improvements

- Improve generalization and reduce overfitting.
- Experiment with different Inception configurations.
- Compare the model with CNN, LSTM, and hybrid architectures.
- Evaluate performance using precision, recall, F1-score, and a confusion matrix.
- Optimize hyperparameters and document experimental results.

## Author

**Umut Emre Özdamar**

Computer Engineering Student

GitHub: [@umutemre12324](https://github.com/umutemre12324)

## Acknowledgements

Thanks to the University of California, Irvine (UCI) Machine Learning Repository for providing the Human Activity Recognition Using Smartphones dataset.
