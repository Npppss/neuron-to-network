from csv import excel

import matplotlib.pyplot as plt
import cupy as cp
import pandas as pd

class Dense:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        # Inisialisasi He untuk performa optimal pada ReLU
        self.W = cp.random.randn(input_size, output_size) * cp.sqrt(2. / input_size)
        self.b = cp.zeros((1, output_size))
        self.lr = learning_rate
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return cp.dot(inputs, self.W) + self.b

    def backward(self, dZ):
        #Menghitung gradien
        self.dW = cp.dot(self.inputs.T, dZ)
        self.db = cp.sum(dZ, axis=0, keepdims=True)
        dX = cp.dot(dZ, self.W.T)
        # Menghitung gradien untuk diteruskan ke layer sebelumnya
        dInputs = cp.dot(dZ, self.W.T)
        # Update parameter Descent
        self.W -= self.lr * self.dW
        self.b -= self.lr * self.db

        return dInputs

class ReLU:
    def __init__(self):
        self.inputs = None

    def forward(self, Z):
        self.inputs = Z
        return cp.maximum(0, Z)

    def backward(self, dA):
        #Turunan dari ReLU
        dZ = dA * (self.inputs > 0)
        return dZ

class Sequential:
    def __init__(self):
        self.layers =[]
        self.history = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, X):
        # Mengalirkan data maju melewati semua layer
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def fit(self, X, Y, epochs=5000):
        m=len(X)
        print("Training dimulai...")

        for epoch in range(epochs):
            # Forward Propagation
            A = self.predict(X)
            # Menghitung loss (Mean Squared Error)
            loss = cp.mean((A - Y) ** 2)
            #htung loss
            self.history.append(loss.get())
            # Backward Propagation
            #Turunan awal dari fungsi MSE: 2 * (Prediksi - Aktual) / jumlah_data
            dA = 2 * (A - Y) / m
            # Mengalirkan gradien mundur melewati semua layer secara terbalik
            for layer in reversed(self.layers):
                dA = layer.backward(dA)

            # Menampilkan loss setiap 5000 epoch
            if (epoch + 1) % 5000 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

        print("Training selesai.")

# Data
cp.random.seed(42)

print("Memuat dataset...")
df=pd.read_excel("data/data.xlsx")

X=cp.array(df[['AT', 'V', 'AP', 'RH']].values)
Y=cp.array(df[['PE']].values)

#Normalisasi data Min-Max
X_min, X_max = cp.min(X, axis=0), cp.max(X, axis=0)
Y_min, Y_max = cp.min(Y, axis=0), cp.max(Y, axis=0)

X_scaled = (X - X_min) / (X_max - X_min)
Y_scaled = (Y - Y_min) / (Y_max - Y_min)

#Proses Train/test split (80% train, 20% test)
m_total = len(X_scaled)
split_idx = int(m_total * 0.8) # Titik potong 80%

#Mengacak data sebelum split
indices = cp.random.permutation(m_total)
train_idx, test_idx = indices[:split_idx], indices[split_idx:]

#memisahkan data train dan test
X_train, Y_train = X_scaled[train_idx], Y_scaled[train_idx]
X_test, Y_test = X_scaled[test_idx], Y_scaled[test_idx]

print(f"Data Training: {len(X_train)} baris")
print(f"Data Testing: {len(X_test)} baris")

# Melatih model
model = Sequential()

#Merakit arsitektur (mirip Keras/PyTorch)
model.add(Dense(input_size=4, output_size=8, learning_rate=0.01))
model.add(ReLU())
model.add(Dense(input_size=8, output_size=1, learning_rate=0.01))
# Output layer untuk regresi tidak ditambahkan aktivasi (linear)


#Mulai melatih model
model.fit(X_train, Y_train, epochs=5000)


print("\n--- Memulai Simulasi Prediksi ---")
# Contoh input kondisi cuaca khas iklim tropis/Bogor:
# Suhu udara (AT): 26.5 °C
# Vakum (V): 50.0 cmHg
# Tekanan udara (AP): 1010.0 mb
# Kelembapan relatif (RH): 88.0%
cuaca_indonesia = cp.array([[26.5, 50.0, 1010.0, 88.0]])

# 1. Normalisasi input menggunakan X_min dan X_max yang sama saat training
cuaca_scaled = (cuaca_indonesia - X_min) / (X_max - X_min)

# 2. Lakukan tebakan dengan melewatkan data ke dalam model
prediksi_scaled = model.predict(cuaca_scaled)

# 3. Denormalisasi hasil (kembalikan ke satuan energi asli MW)
prediksi_asli = prediksi_scaled * (Y_max - Y_min) + Y_min

# Tampilkan hasil (Gunakan .get() untuk menarik data dari memori GPU)
print(f"Prediksi Output Energi (PE) untuk kondisi lokal: {prediksi_asli.get()[0][0]:.2f} MW")

# Evaluasi akhir & R-Squared

print("\n Evaluasi model pada Data Testing ")
#Menyuruh model menebak soal yang belum pernah diliat
prediksi_test=model.predict(X_test)

#menghitung loss ujian
test_loss=cp.mean((prediksi_test-Y_test)**2)
print(f"Loss (MSE) pada Data Ujian: {test_loss.get():.6f}")

#Menghitung R-Squared (akurasi presentase)
SS_res = cp.sum((Y_test - prediksi_test) ** 2)
SS_tot = cp.sum((Y_test - cp.mean(Y_test)) ** 2)
r2_score = 1 - (SS_res / SS_tot)

print(f"Akurasi Prediksi Model (R-Squared): {r2_score.get() * 100:.2f}%")

# MENYIMPAN BOBOT MODEL (SAVE WEIGHTS)
print("\n--- Menyimpan Otak Model ---")
# Mengambil bobot (W) dan bias (b) dari layer Dense pertama dan kedua
W1_simpan = model.layers[0].W
b1_simpan = model.layers[0].b
W2_simpan = model.layers[2].W
b2_simpan = model.layers[2].b

# Menyimpan ke dalam file biner CuPy (.npy)
cp.save('W1_model.npy', W1_simpan)
cp.save('b1_model.npy', b1_simpan)
cp.save('W2_model.npy', W2_simpan)
cp.save('b2_model.npy', b2_simpan)

print("Berhasil! Semua ingatan model telah disimpan ke file .npy")

plt.plot(model.history)
plt.title('Grafik Penurunan Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.show()