import numpy as np
import torch
import torch.nn as nn

print("===MEMBANGUN MLP DENGAN PYTORCH===\n")

#setiap model PyTorch WAJIB mewarisi (inherit) dari nn.Module
class IrisMLPPyTorch(nn.Module):
        def __init__(self, n_input, n_hidden, n_output):
        #memanggil inisialisasi dari nn.Module (Wajib)
            super(IrisMLPPyTorch, self).__init__()
            #nn.Linear otomatis membuatkan matriks 'weights' dan 'bias' untukmu!
            #Ia juga secara otomatis mengatur ukurannya agar tidak terjadi error dimensi matriks.
            self.hidden_layer = nn.Linear(n_input, n_hidden)
            self.output_layer = nn.Linear(n_hidden, n_output)

        def forward(self, x):
            #forward pass sangat sederhana. Panggil layernya, masukkan ke fungsi aktivasi.
            #torch.relu dan torch.softmax sudah tersedia bawaan!
            z_hidden = self.hidden_layer(x)
            a_hidden = torch.relu(z_hidden) #Kita coba gunakan ReLU sebagai variasi
            
            z_output = self.output_layer(a_hidden)
            #Catatan: Di PyTorch, Softmax sering kali digabung dengan Loss Function nanti,
            #jadi kita cukup mengembalikan nilai mentah (z_output) atau yang sering disebut "Logits".
            return z_output
        

# Mari kita cetak "blueprint" robot baru kita
# 4 Input (Fitur Bunga) -> 5 Hidden -> 3 Output (Kelas Spesies)
model = IrisMLPPyTorch(4, 5, 3)

print("Blueprint Model:\n", model)

# Mengintip parameter (bobot dan bias) yang di-generate otomatis
print("\nBobot di Hidden Layer (Otomatis dibuat):")
print(model.hidden_layer.weight)

# Memindahkan seluruh model ke GPU hanya dengan SATU baris!
if torch.cuda.is_available():
    model = model.to('cuda')
    print("\n✅ Model berhasil dipindahkan ke RTX 4050 (GPU)!")