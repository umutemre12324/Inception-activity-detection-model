import numpy as np
import matplotlib.pyplot as plt
import joblib 
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

x_test_scaled = np.load("x_test_scaled.npy")
y_test_to = np.load("y_test_to.npy")

model = load_model("best_model.h5")
y_pred = model.predict(x_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_to, axis=1)

activity_labels= {
    0: "WALKING",   
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING"
}


