import numpy as np
import torch

print("==FASE 1: SINTAKS DASAR (NUMPY VS PYTORCH)==\n")

## Membuat matriks 2x2
matriks_np = np.array([[1, 2], [3, 4]])
matriks_pt = torch.tensor([[1, 2], [3, 4]])

print("NumPy Array:\n", matriks_np)
print("\nPyTorch Tensor:\n", matriks_pt)
print("-" * 50)

print("==FASE 2: OPERASI MATRIKS (Dot Product)==\n")
#mengkalikan matriks 2x2 dengan dirinya sendiri 
hasil_np = np.dot(matriks_np, matriks_np)
#di PyTorch, kita bisa menggunakan fungsi torch.mm() atau operator @
hasil_pt=matriks_pt @ matriks_pt

print("Hasil Dot Product (NumPy):\n", hasil_np)
print("\nHasil Dot Product (PyTorch):\n", hasil_pt)
print("-" * 50)

print("==FASE 3: KEKUATAN GPU (CUDA)==\n")
# Mengecek apakah sistem komputermu memiliki GPU Nvidia yang kompatibel
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU Terdeteksi! Memindahkan Tensor ke GPU...")
    tensor_gpu = matriks_pt.to(device)
    print("Lokasi Tensor saat ini:", tensor_gpu.device)
else:
    print("GPU tidak terdeteksi. Tensor tetap menggunakan CPU.")
print("-" * 50)

print("==FASE 4: KEAJAIBAN AUTOGRAD (SELAMAT TINGGAL KALKULUS MANUAL)==\n")
#Mari kita buat sebuah fungsi matematika sederhana: y = x^2 + 3x
#Secara manual (kalkulus), turunan dy/dx adalah: 2x + 3
#Jika kita set nilai x = 2.0, maka turunan gradiennya harusnya: 2(2) + 3 = 7.0

#requires_grad=True adalah saklar ajaib yang menyuruh PyTorch melacak perhitungan kalkulusnya
x = torch.tensor(2.0, requires_grad=True) 

# Lakukan Forward Pass (Operasi Matematika)
y=(x ** 2)+(3 * x)
print(f"Nilai x={x.item()}")
print(f"Hasil fungsi y = x^2 + 3x adalah {y.item()}")

# Lakukan Backward Pass (Menghitung Gradien)
y.backward()

# Lihat hasilnya!
print(f"Gradien (dy/dx) yang dihitung otomatis oleh PyTorch: {x.grad.item()}")
print("Perhitungan cocok dengan kalkulus manual (7.0)!")