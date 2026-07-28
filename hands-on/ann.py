import cupy as cp
import pandas as pd

file_name='data/data.xlsx'

df=pd.read_excel(file_name)

## Memisahkan Fitur X dan Target Y
X=cp.array(df[['AT', 'V', 'AP', 'RH']].values)
Y=cp.array(df[['PE']].values)

## Normalisasi Data
X_min, X_max= X.min(axis=0), X.max(axis=0)
Y_min, Y_max= Y.min(axis=0), Y.max(axis=0)

X_scaled= (X - X_min) / (X_max - X_min)
Y_scaled= (Y - Y_min) / (Y_max - Y_min)

## Inisialisasi Bobot dan Bias
input_size=4
hidden_size=8
output_size=1
learning_rate=0.01
epochs=5000

m=len(X_scaled)

# Menggunakan inisialisasi He untuk ReLU dengan CuPy
cp.random.seed(42)
W1=cp.random.randn(input_size, hidden_size) * cp.sqrt(2. / input_size)
b1=cp.zeros((1, hidden_size))
W2=cp.random.randn(hidden_size, output_size) * cp.sqrt(2. / hidden_size)
b2=cp.zeros((1, output_size))

# Proses Training
for epoch in range(epochs):
    ## forward propagation
    Z1=cp.dot(X_scaled, W1) + b1
    A1=cp.maximum(0, Z1)  # ReLU activation

    Z2=cp.dot(A1, W2) + b2
    A2=Z2  # Linear activation for output layer

    #menghitung loss (Mean Squared Error)
    loss=cp.mean((A2 - Y_scaled) ** 2)  

    ##Backpropagation
    dA2=2 * (A2 - Y_scaled) / m
    dW2=cp.dot(A1.T, dA2)
    db2=cp.sum(dA2, axis=0, keepdims=True)

    dA1=cp.dot(dA2, W2.T)
    dZ1=dA1 * (Z1 > 0)  # Derivative
    dw1=cp.dot(X_scaled.T, dZ1)
    db1=cp.sum(dZ1, axis=0, keepdims=True)

    #update parameter
    W1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss (MSE): {loss.get():.6f}")


print("Training selesai.")



# Kondisi cuaca lokal (Contoh: Bogor dengan suhu hangat dan lembap)
lokasi_indonesia = cp.array([[28.0, 60.0, 1010.0, 85.0]])

# Normalisasi menggunakan parameter dari GPU
lokasi_scaled = (lokasi_indonesia - X_min) / (X_max - X_min)

# Forward pass di GPU
Z1_pred = cp.dot(lokasi_scaled, W1) + b1
A1_pred = cp.maximum(0, Z1_pred)
Z2_pred = cp.dot(A1_pred, W2) + b2

# Kembalikan ke nilai asli (Megawatt)
prediksi_energi = Z2_pred * (Y_max - Y_min) + Y_min

# Pindahkan hasil akhir kembali ke CPU (.get()) untuk ditampilkan atau disimpan
print(f"Prediksi Output Energi Listrik (PE): {prediksi_energi.get()[0][0]:.2f} MW")