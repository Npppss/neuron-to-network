import numpy as np

class MultiLayerPerceptron:
    def __init__(self, n_input,n_hidden,n_output):
        # 1. Koneksi Input -> Hidden Layer
        # Bentuk Matriks: (2 baris x 3 kolom)
        self.weights_input_hidden = np.random.rand(n_input, n_hidden)*0.1  # Inisialisasi bobot dengan nilai acak kecil
        self.bias_hidden=np.zeros((1,n_hidden))# Bias untuk 3 neuron

        # 2. Koneksi Hidden Layer -> Output Layer
        # Bentuk Matriks: (3 baris x 1 kolom)
        self.weights_hidden_output=np.random.randn(n_hidden,n_output)*0.1  #inisialisasi bobot dengan nilai acak kecil
        self.bias_output=np.zeros((1,n_output))  # Bias untuk 1 neuron output

    def sigmoid(self,z):
            return 1/(1+np.exp(-z))
        
    def forward_pass(self, inputs):
            # PERJALANAN DATA

            # Stasiun 1: Masuk ke Hidden Layer
            z_hidden=np.dot(inputs,self.weights_input_hidden) + self.bias_hidden
            a_hidden=self.sigmoid(z_hidden)# Output dari 3 neuron tersembunyi

            # Stasiun 2: Masuk ke Output Layer
            z_output=np.dot(a_hidden,self.weights_hidden_output) + self.bias_output
            a_output=self.sigmoid(z_output) # Output akhir (Tebakan)

            return a_hidden, a_output
        

        #Uji Coba Arsitektur

# Data dummy: 4 orang, masing-masing dengan 2 fitur (Jam Belajar, Jam Tidur)
data_masuk = np.array([
    [2.0, 8.0],
    [5.0, 6.0],
    [8.0, 7.0],
    [1.0, 4.0]
])

#Membangun Jaringan: 2 Input -> 3 Hidden -> 1 Output
jaringan_saraf=MultiLayerPerceptron(n_input=2, n_hidden=3, n_output=1)

## Lakukan satu kali tebakan (Forward Pass)
hasil_hidden,tebakan_akhir=jaringan_saraf.forward_pass(data_masuk)

print("=== BENTUK DATA YANG MENGALIR ===")
print(f"Bentuk Input Asli: {data_masuk.shape} (4 Data, 2 Fitur)")
print(f"Bentuk di Hidden Layer: {hasil_hidden.shape} (4 Data diproses 3 Neuron)")
print(f"Bentuk Tebakan Akhir: {tebakan_akhir.shape} (4 Data dengan 1 Peluang)\n")

print("=== HASIL TEBAKAN MENTAH ===")
print(tebakan_akhir)