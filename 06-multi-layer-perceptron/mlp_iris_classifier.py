import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## Fase 1: Load Dataset Iris    
print("Memuat Dataset Iris...")
iris=load_iris()
X=iris.data #Fitur (Panjang/Lebar Kelopak & Mahkota)
y=iris.target #Label asli (0, 1, 2)

# Filter hanya 2 kelas pertama (0=Setosa, 1=Versicolor) agar menjadi Klasifikasi Biner
mask=y<2
X_biner=X[mask]
y_biner=y[mask].reshape(-1,1) #Ubah bentuk array Y menjadi matriks kolom tegak

print(f"Data siap: {X_biner.shape[0]} baris dengan {X_biner.shape[1]} fitur.")

#Memecahkan data: 80% untuk pelatihan, 20% untuk pengujian
X_train,X_test,y_train,y_test=train_test_split(X_biner,y_biner,test_size=0.2,random_state=42)

# Feature Scaling (Standardisasi)
# Menyamakan skala angka (mean=0, variance=1) agar model tidak bias pada angka besar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)#Ingat: Testing set hanya di-transform, tidak di-fit!

class MultiLayerPerceptron:
    def __init__(self,n_input,n_hidden,n_output):
        # Inisialisasi Matriks Bobot & Bias
        self.weights_hidden=np.random.randn(n_input,n_hidden) # Bobot Input -> Hidden
        self.bias_hidden=np.zeros((1,n_hidden)) # Bias Hidden Layer

        self.weights_output=np.random.randn(n_hidden,n_output) # Bobot Hidden -> Output
        self.bias_output=np.zeros((1,n_output)) # Bias Output Layer

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def turunan_sigmoid(self,a):
        return a*(1-a)
    
    def forward_pass(self,inputs):
        # Layer 1: Hidden
        z_hidden=np.dot(inputs,self.weights_hidden)+self.bias_hidden
        a_hidden=self.sigmoid(z_hidden)

        # Layer 2: Output
        z_output=np.dot(a_hidden,self.weights_output)+self.bias_output
        a_output=self.sigmoid(z_output)

        return a_hidden,a_output
    
    def train(self,inputs,target,learning_rate,epochs):
        print("Memulai Pelatihan Multi-Layer Perceptron...\n")

        for epoch in range(epochs):
            # Forward Pass
            a_hidden,a_output=self.forward_pass(inputs)

            # Hitung Loss (Mean Squared Error)
            error=a_output-target
            loss=np.mean(error**2)

            # Backpropagation

            # --- Fase 1: Mundur dari Output Layer ---
            d_output=error*self.turunan_sigmoid(a_output) # Turunan Loss terhadap Output
            dw_output=np.dot(a_hidden.T,d_output) # Turunan Bobot Hidden -> Output
            #Menjumlahkan error bias untuk semua baris data
            db_output=np.sum(d_output,axis=0,keepdims=True) # Turunan Bias Output

            # --- Fase 2: Mundur ke Hidden Layer ---
            # Melempar error dari output kembali ke hidden layer
            error_hidden=np.dot(d_output,self.weights_output.T)
            d_hidden=error_hidden*self.turunan_sigmoid(a_hidden) # Turunan Loss terhadap Hidden
            dw_hidden=np.dot(inputs.T,d_hidden) # Turunan Bobot Input -> Hidden
            db_hidden=np.sum(d_hidden,axis=0,keepdims=True) # Turunan Bias Hidden

            # 4. GRADIENT DESCENT (Update Bobot & Bias)
            self.weights_output-=learning_rate*dw_output
            self.bias_output-=learning_rate*db_output
            self.weights_hidden-=learning_rate*dw_hidden
            self.bias_hidden-=learning_rate*db_hidden

            #cetak Rapor
            if (epoch + 1) % 1000 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:5d} | Loss: {loss:.6f}")

#FASE 3: EKSEKUSI & EVALUASI
model=MultiLayerPerceptron(n_input=4,n_hidden=5,n_output=1) #4 fitur input, 5 neuron hidden, 1 neuron output
#Latih HANYA menggunakan data Training
model.train(X_train_scaled,y_train,learning_rate=0.1,epochs=20000)
print("\n3. Menguji Model pada Data Testing")

# Lakukan Forward Pass pada data yang belum pernah dilihat model sebelumnya
_,probabilitas_prediksi=model.forward_pass(X_test_scaled)

# Ubah probabilitas desimal menjadi keputusan tegas (0 atau 1) dengan batas 0.5
keputusan_final=(probabilitas_prediksi > 0.5).astype(int)

# Hitng Akurasi: Bandingkan keputusan model dengan label asli
akurasi=np.mean(keputusan_final==y_test)*100

print(f"\n========================================")
print(f"Akurasi Model pada Data Ujian: {akurasi:.2f}%")
print(f"========================================")