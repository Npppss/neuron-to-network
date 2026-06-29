import numpy as np

class NonLinearNeuron:
    def __init__(self, jumlah_input):
        # Initialize weights and bias randomly
        self.weights = np.random.randn(jumlah_input) * 0.1
        self.bias = 0.0

    def sigmoid(self, z):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-z))
    
    #Rumus turunan dari fungsi sigmoid
    def sigmoid_turunan(self, z):
        return 1 / (1+np.exp(-z))
    
    def forward_pass(self, inputs):
        kombinasi_linear = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(kombinasi_linear)
    
    def train(self, inputs, jawaban_asli, learning_rate, epochs):
        print("Memulai pelatihan dengan Fungsi Aktivasi...")
        print("-----------------------------------------")
        for epoch in range(epochs):
            # Forward pass
            tebakan_neuron = self.forward_pass(inputs)
            # Calculate loss (Mean Squared Error)
            error = tebakan_neuron - jawaban_asli
            loss = np.mean(error ** 2)
            # Backpropagation dengan Chain Rule

            ## GIGI 1: Turunan dari Error (berapa jarak melesetnya)
            dcost_dpred=error
            ## GIGI 2: Turunan dari Fungsi Aktivasi Sigmoid
            dpred_dz=self.sigmoid_turunan(tebakan_neuron)

            # Gabungkan Gigi 1 dan Gigi 2 (Ini disebut "Delta")
            delta=dcost_dpred*dpred_dz

            #GIGI 3: Kalikan dengan Input untuk mencari turunan Bobot (dw) dan Bias (db)
            dw=(1/len(inputs))*np.dot(inputs.T,delta)
            db=(1/len(inputs))*np.sum(delta)

            #Update weights and bias
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            if (epoch+1) % 1000 == 0 or epoch == 0:
                print(f"Epoch ke-{epoch+1}/{epochs} | Loss: {loss:.6f}")


# Data: Lama belajar dalam jam (Fitur X)
data_belajar = np.array([[1.0], [2.0], [6.0], [8.0]])

# Target Y: 0 (Gagal Ujian) dan 1 (Lulus Ujian)
# Karena pakai Sigmoid, target HARUS antara 0 dan 1
target_lulus = np.array([0, 0, 1, 1])

neuron_klasifikasi = NonLinearNeuron(jumlah_input=1)

# Kita butuh epoch lebih banyak (10.000) dan learning rate lebih besar
# karena kurva Sigmoid butuh waktu lebih lama untuk bergeser
neuron_klasifikasi.train(data_belajar, target_lulus, learning_rate=0.5, epochs=10000)

print("\n-------------------")
print("Uji Coba Prediksi Baru!")
uji_coba = np.array([[6.0]]) # Jika belajar 7 jam, lulus tidak?
hasil = neuron_klasifikasi.forward_pass(uji_coba)
print(f"Peluang lulus jika belajar 7 jam: {hasil[0]*100:.2f}%")