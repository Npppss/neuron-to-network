# 🧮 Deep Learning Math Cheat Sheet

Catatan ini merangkum seluruh arsitektur matematika fundamental yang digunakan untuk membangun *Artificial Neural Networks* (ANN) dari nol menggunakan NumPy di dalam repositori ini.

---

## 1. Operasi Linear (Forward Pass)
Kombinasi linear antara input ($X$), bobot ($W$), dan bias ($b$). Ini adalah jantung komputasi dari setiap neuron.

$$Z = X \cdot W + b$$

> **Keterangan:**
> * $\cdot$ mewakili perkalian matriks (*dot product*).

---

## 2. Fungsi Aktivasi
Berfungsi untuk membengkokkan ruang data linear sehingga model dapat mengenali pola non-linear yang kompleks.

### Sigmoid
Digunakan untuk probabilitas klasifikasi biner dan *hidden layer*. Menekan nilai *output* ke rentang $0$ hingga $1$.

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### Turunan Sigmoid (Derivative)
Digunakan pada fase *Backpropagation* untuk menghitung seberapa tajam gradien di titik aktivasi tersebut. (Diasumsikan $a$ adalah hasil *output* dari fungsi Sigmoid).

$$\sigma'(a) = a \cdot (1 - a)$$

### Softmax
Digunakan pada *Output Layer* untuk klasifikasi multi-kelas. Mengubah keluaran beberapa neuron menjadi distribusi probabilitas yang jika dijumlahkan totalnya tepat $1.0$ ($100\%$).

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

---

## 3. Fungsi Evaluasi / Loss Functions
Metrik untuk menghitung seberapa jauh tebakan model (prediksi) meleset dari jawaban yang sebenarnya (target).

### Mean Squared Error (MSE)
Digunakan umumnya untuk kasus regresi atau eksperimen jaringan dasar.

$$L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Categorical Cross-Entropy
Digunakan berpasangan dengan fungsi Softmax. Memberikan penalti logaritmik yang besar jika model sangat yakin tetapi salah menebak.

$$L = - \sum_{c=1}^{M} y_c \log(\hat{y}_c)$$

> **Keterangan:**
> * $y$ = Target asli (*One-Hot Encoded*)
> * $\hat{y}$ = Probabilitas prediksi dari model

---

## 4. Backpropagation & Aturan Rantai (Chain Rule)
Proses mendistribusikan kesalahan (*error*) dari *Output Layer* kembali menelusuri *Hidden Layer* untuk menghitung gradien.

### Gradien Output (Kombinasi Softmax + Cross-Entropy)
Penurunan fungsi Softmax dan Cross-Entropy saling menyederhanakan, menyisakan rumus selisih murni yang sangat elegan:

$$dZ_{\text{output}} = \hat{y} - y$$

### Gradien Bobot (Weight Gradient)
Menghitung seberapa besar bobot harus dikoreksi berdasarkan letak kesalahan.

$$dW = X^T \cdot dZ$$

### Melempar Error ke Belakang (Hidden Layer Error)
Menggunakan matriks transposisi ($T$) dari bobot *layer* di depannya untuk mengalokasikan persentase kesalahan ke neuron sebelumnya.

$$\text{Error}_{\text{hidden}} = dZ_{\text{output}} \cdot W_{\text{output}}^T$$

---

## 5. Algoritma Optimasi (Gradient Descent)
Langkah terakhir di penghujung *epoch* untuk memperbarui nilai parameter secara simultan menggunakan nilai *Learning Rate* ($\alpha$).

### Update Bobot (Weights)
$$W = W - \alpha \cdot dW$$

### Update Bias
$$b = b - \alpha \cdot db$$