import keras
import numpy as np
from keras.layers import Dense
from sklearn.model_selection import train_test_split


np.random.seed(7)
dataset = np.loadtxt(r"C:\Users\KADİR İNCE\OneDrive\Resimler\Masaüstü\makina_ögrenmesi.final\2025_Veriseti.csv", delimiter=",")

inputs = dataset[:, 0:10]
outputstemp = dataset[:, 10]
outputs = []
for i in outputstemp:
    t = [0, 0, 0]
    t[int(i)] = 1
    outputs.append(t)

inputs = np.array(inputs)
outputs = np.array(outputs)
train_data, test_data, train_labels, test_labels = train_test_split(inputs,outputs, test_size=0.2, random_state=42)

model = keras.models.Sequential(
    [
        Dense(512, input_shape=(10,) ,activation='relu'),
        Dense(256, activation='relu'),
        Dense(3, activation='tanh')
    ]
)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=1000, batch_size=150)

train_basarim = model.evaluate(train_data, train_labels)
print("\nEğitim Başarımı: %s : %.2f%%" % (model.metrics_names[1], train_basarim[1] * 100))

# Test başarımını değerlendirme
test_basarim = model.evaluate(test_data, test_labels)
print("\nTest Başarımı: %s : %.2f%%" % (model.metrics_names[1], test_basarim[1] * 100))