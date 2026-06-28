import numpy as np

class IntelligentNeuron:
    def __init__(self,jumlah_input):
        #inisialisasi bobot dan bias secara acak
        self.weights=np.random.randn(jumlah_input)*0.1
        self.bias=0.0

    def forward_pass(self,input):
        #tebakan linear mentah nya
        return np.dot(input,self.weights)+self.bias
    
    def train(self,input,jawaban_asli,learning_rate,epochs):
        print()
        for epoch in range(epochs):
            #melakukan forward pass
            tebakan_neuron=self.forward_pass(input)
            #menghitung loss & loss MSE
            error=tebakan_neuron-jawaban_asli
            loss=np.mean(error**2)
            #Backpropagation untuk menghitung gradien
            dw=(2/len(input))*np.dot(input.T,error)
            db=(2/len(input))*np.sum(error)
            #Gradien descent untuk memperbarui bobot dan bias
            self.weights-=learning_rate*dw
            self.bias-=learning_rate*db
            #cetak perkemabangan loss setiap 100 epoch
            if (epoch+1)%100==0 or epoch==0:
                print(f"Latihan ke-{epoch+1} | Loss (Tingkat Kesalahan): {loss:.6f}")

#kita punya 4 data input dan 1 output target
#variabel bebas (X) dan variabel target (Y)
data_input = np.array([[1.0], 
                       [2.0], 
                       [3.0], 
                       [4.0]])

##Jawaban asli (target) yang diharapkan harga asli nya 2x lipat jumalah kamar
#jadi target ideal kita adalah memprediksi angka: [2.0, 4.0, 6.0, 8.0]
target_asli=np.array([2.0,4.0,6.0,8.0])

## buat neuron cerdas dengan 4 input
neuron_cerdas=IntelligentNeuron(jumlah_input=1)

#suruh neuron cerdas untuk belajar dari data input dan target asli
neuron_cerdas.train(data_input,target_asli,learning_rate=0.01,epochs=1000)

print("\n----------------")
print("Pelatihan selesai")
print(f"Bobot akhir yang dipelajari neuron (Harusnya mendekati 2): {neuron_cerdas.weights[0]:.4f}")
print(f"Bias akhir yang dipelajari neuron (Harusnya mendekati 0): {neuron_cerdas.bias:.4f}")