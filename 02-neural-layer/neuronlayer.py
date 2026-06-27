import numpy as np

class NeuronLayer:
    def __init__(self, jumlah_neuron, jumlah_input):
        ##Initialize the weights and bias for each neuron in the layer
        self.weights = np.random.rand(jumlah_neuron, jumlah_input)
        self.bias = np.random.rand(jumlah_neuron)

    def fungsi_aktivasi(self, x):
        ##Sigmoid activation function dari 0-1
        return 1 / (1 + np.exp(-x))
    
    def forward_pass(self, inputs):
        ##Calculate the weighted sum of inputs and biases for each neuron
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        ##Apply the activation function to each neuron's output
        output = self.fungsi_aktivasi(weighted_sum)
        return output
    
# Create an instance of the NeuronLayer class with 3 neurons and 4 inputs
neuron_layer_saya = NeuronLayer(jumlah_neuron=3, jumlah_input=4)

data_masuk = np.array([1.5, 2.0, 0.5, 1.0])  # Example input data with 4 features

hasil_prediksi_layer = neuron_layer_saya.forward_pass(data_masuk)
print(f"Hasil output layer neuron: {hasil_prediksi_layer}")