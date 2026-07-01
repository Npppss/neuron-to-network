import numpy as np
import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. PERSIAPAN DATA (MULTI-CLASS)
iris=load_iris()
X=iris.data
y_asli=iris.target #Label asli (0, 1, 2)

#Menerapkan One-Hot Encoding untuk label multi-class
jumlah_kelas=3
y_one_hot=np.eye(jumlah_kelas)[y_asli] #Mengubah label menjadi matriks one-hot

X_train,X_test,y_train,y_test=train_test_split(X,y_one_hot,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test) #Hanya transform, tidak fit!

# ARSITEKTUR KELAS MLP (UPDATED)
class MultiClassMLP:
    def __init__(self,n_input,n_hidden,n_output):
        self.weights_hidden=np.random.randn(n_input,n_hidden)
        self.bias_hidden=np.zeros((1,n_hidden))
        self.weights_output=np.random.randn(n_hidden,n_output)
        self.bias_output=np.zeros((1,n_output))

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def turunan_sigmoid(self,a):
        return a*(1-a)
    
    def softmax(self,z):
        # Mencegah Overflow dengan mengurangi nilai maksimum dari z
        ekspo=np.exp(z-np.max(z,axis=1,keepdims=True))
        return ekspo/np.sum(ekspo,axis=1,keepdims=True)
    
    def forward_pass(self,inputs):
        z_hidden=np.dot(inputs,self.weights_hidden)+self.bias_hidden
        a_hidden=self.sigmoid(z_hidden)

        z_output=np.dot(a_hidden,self.weights_output)+self.bias_output
        a_output=self.softmax(z_output)

        return a_hidden,a_output
    
    def train(self,inputs,target,learning_rate,epochs):
        print("Memulai Pelatihan Multi-Class MLP...\n")
        n_samples=inputs.shape[0]

        for epoch in range(epochs):
            a_hidden,a_output=self.forward_pass(inputs)

            #Hitung Categorical Cross-Entropy Loss (tambah 1e-9 agar tidak error log 0)
            loss=-np.sum(target*np.log(a_output+1e-9))/n_samples

            # Backpropagation
            # Gradient dari Loss terhadap Output
            d_output=(a_output-target)/n_samples

            dw_output=np.dot(a_hidden.T,d_output)
            db_output=np.sum(d_output,axis=0,keepdims=True)

            error_hidden=np.dot(d_output,self.weights_output.T)
            d_hidden=error_hidden*self.turunan_sigmoid(a_hidden)

            dw_hidden=np.dot(inputs.T,d_hidden)
            db_hidden=np.sum(d_hidden,axis=0,keepdims=True)

            self.weights_output-=learning_rate*dw_output
            self.bias_output-=learning_rate*db_output
            self.weights_hidden-=learning_rate*dw_hidden
            self.bias_hidden-=learning_rate*db_hidden

            if (epoch + 1) % 1000 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:5d} | Loss: {loss:.6f}")

#3. EKSEKUSI & EVALUASI
#4 Input (Fitur) -> 5 Hidden Neuron -> 3 Output Neuron (Kelas Spesies)
model=MultiClassMLP(n_input=4,n_hidden=5,n_output=3)
model.train(X_train_scaled, y_train, learning_rate=0.5, epochs=5000)

#Uji pada Data Test
_,prediksi_test=model.forward_pass(X_test_scaled)

#Ambil indeks dengan probabilitas paling besar menggunakan argmax
kelas_tebakan=np.argmax(prediksi_test,axis=1)
kelas_asli=np.argmax(y_test,axis=1)

akurasi=np.mean(kelas_tebakan==kelas_asli)*100
print(f"Akurasi Model (3 Kelas): {akurasi:.2f}%")