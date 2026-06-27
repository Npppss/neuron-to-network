import numpy as np

class Neuron1:
    def __init__(self, jumlah_input):
        ##Initialize the weights and bias
        self.weights = np.random.rand(jumlah_input)
        self.bias = np.random.rand(1)

    def fungsi_aktivasi(self, x):
        ##Sigmoid activation function dari 0-1
        return 1 / (1 + np.exp(-x))
    
    def forward_pass(self, inputs):
        ##Calculate the weighted sum of inputs and bias
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        ##Apply the activation function
        output = self.fungsi_aktivasi(weighted_sum)
        return output

# Create an instance of the Neuron1 class with 4 inputs
neuron_saya=Neuron1(jumlah_input=3)

# Masukkan array dengan 4 angka (contoh: tinggi, berat, umur, detak jantung)
data_masuk= np.array([1.5, 2.0, 0.5])

# Menjalankan neuron
hasil_prediksi=neuron_saya.forward_pass(data_masuk)
    
print(f"Hasil output neuron: {float(hasil_prediksi):.4f}")