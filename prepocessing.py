
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import joblib

# Verileri yükle
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

# Etiketleri one-hot encoding formatına dönüştür
number_of_classes = len(np.unique(y_train))
y_train_to = to_categorical(y_train, num_classes=number_of_classes)
y_test_to = to_categorical(y_test, num_classes=number_of_classes)

# Her sinyal kanalı için ayrı scaler oluştur
scalers = {}

x_train_scaled = np.zeros_like(x_train, dtype=np.float32)
x_test_scaled = np.zeros_like(x_test, dtype=np.float32)

for i in range(x_train.shape[-1]):
    scaler = StandardScaler()

    # Scaler yalnızca eğitim verisiyle öğrenilir
    x_train_scaled[:, :, i] = scaler.fit_transform(x_train[:, :, i])
    x_test_scaled[:, :, i] = scaler.transform(x_test[:, :, i])

    scalers[i] = scaler

# Ölçeklenmiş verileri kaydet
np.save("x_train_scaled.npy", x_train_scaled)
np.save("x_test_scaled.npy", x_test_scaled)
np.save("y_train_to.npy", y_train_to)
np.save("y_test_to.npy", y_test_to)

# Scaler nesnelerini kaydet
joblib.dump(scalers, "scaler.pkl")

print("Veri ön işleme tamamlandı!")
print("Eğitim verisi:", x_train_scaled.shape)
print("Test verisi:", x_test_scaled.shape)
print("Eğitim etiketleri:", y_train_to.shape)
print("Test etiketleri:", y_test_to.shape)
