import numpy as np

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


# Simulasi: Menyelesaikan Gerbang Logika XOR
# XOR adalah masalah klasik yang TIDAK BISA diselesaikan oleh 1 neuron biasa.
# Hanya Multi-Layer Perceptron yang bisa memecahkannya.

# Input: 2 kombinasi saklar (0=Mati, 1=Nyala)
data_masuk = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target XOR: Lampu nyala (1) HANYA JIKA saklar berbeda
target_asli = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Arsitektur: 2 Input -> 3 Hidden -> 1 Output
jaringan_saraf = MultiLayerPerceptron(n_input=2, n_hidden=3, n_output=1)

# Latih jaringannya (Butuh epoch banyak karena kompleksitas XOR)
jaringan_saraf.train(data_masuk, target_asli, learning_rate=0.5, epochs=10000)

print("\nHASIL PREDIKSI XOR SETELAH TRAINING")
_, hasil_akhir = jaringan_saraf.forward_pass(data_masuk)
for i in range(len(data_masuk)):
    print(f"Input: {data_masuk[i]} | Prediksi Model: {hasil_akhir[i][0]:.4f} | Target Asli: {target_asli[i][0]}")