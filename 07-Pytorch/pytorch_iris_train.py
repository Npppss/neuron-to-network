import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#PERSIAPAN DATA (Numpy ke Tensor)
iris=load_iris()
X=iris.data
y =iris.target  # Di PyTorch, tidak perlu repot One-Hot Encoding manual!

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler=StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled =scaler.transform(X_test)

# Konversi array NumPy menjadi Tensor PyTorch
X_train_tensor =torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor=torch.tensor(y_train, dtype=torch.long)
X_test_tensor =torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor= torch.tensor(y_test, dtype=torch.long)

#pindahkan data ke GPU jika tersedia
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor=X_train_tensor.to(device)
y_train_tensor =y_train_tensor.to(device)
X_test_tensor  = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

print(f"Menggunakan perangkat komputasi: {device}\n")

# 2. ARSITEKTUR MODEL
class IrisMLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(IrisMLP, self).__init__()
        self.hidden = nn.Linear(n_input, n_hidden)
        self.output = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x) # Output Logits (tanpa Softmax, karena akan ditangani oleh Loss Function)
        return x

model=IrisMLP(4, 5, 3).to(device)


# PELATIHAN (TRAINING LOOP)
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.01)

epochs =1000
for epoch in range(epochs):
    # a. Nol-kan gradien sisa dari putaran sebelumnya
    optimizer.zero_grad()
    
    # b. Forward Pass: Minta model menebak
    prediksi = model(X_train_tensor)
    
    # c. Hitung Loss (Seberapa meleset?)
    loss = criterion(prediksi, y_train_tensor)
    
    # d. Backward Pass: Hitung kalkulus (Autograd)
    loss.backward()
    
    # e. Update Bobot (Gradient Descent)
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:4d} | Loss: {loss.item():.6f}")

# EVALUASI AKHIR

# Matikan pelacakan gradien karena kita hanya ingin menebak (menghemat memori)
with torch.no_grad():
    prediksi_test = model(X_test_tensor)
    #Ambil indeks dengan nilai terbesar sebagai tebakan kelas
    kelas_tebakan=torch.argmax(prediksi_test, dim=1)
    
    akurasi=(kelas_tebakan == y_test_tensor).sum().item() / y_test_tensor.size(0) * 100
    print(f"\n========================================")
    print(f"Akurasi PyTorch (3 Kelas): {akurasi:.2f}%")
    print(f"========================================")