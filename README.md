# Building Neural Networks from Scratch with Python

Repositori ini adalah dokumentasi perjalanan pembelajaran saya dalam membedah, memahami, dan membangun **Artificial Neural Networks (ANN)** langsung dari dasar (*from scratch*). 

Alih-alih menggunakan *high-level framework* seperti PyTorch atau TensorFlow, tujuan utama proyek ini adalah menerapkan matematika dasar di balik kecerdasan buatan menggunakan **Python** dan **NumPy**.

## Tech Stack & Teori Dasar
* **Bahasa Pemrograman:** Python 3.8
* **Komputasi Matriks:** NumPy (untuk efisiensi operasi *dot product* dan manipulasi matriks)
* **Konsep Matematika yang Dipelajari:** Aljabar Linear (Matriks), Kalkulus (Kalkulasi Gradien/Turunan), Statistik dasar.

---

## Peta Pembelajaran & Progres Eksperimen

Proyek di dalam repositori ini disusun secara bertahap, mulai dari komponen terkecil hingga membentuk jaringan saraf yang utuh:

### Bagian 1: Fondasi Neuron Tunggal (Perceptron)
* **Single Neuron Simulation:** Membuat satu neuron buatan dasar yang menerima input fitur, mengalikannya dengan bobot (*weights*), menambahkan *bias*, dan melewatkannya ke fungsi aktivasi.
* **Exploration of Activation Functions:** Eksperimen manual dengan fungsi **Sigmoid** ($1 / (1 + e^{-x})$) dan memahami fenomena kejenuhan nilai (*saturation*), serta alternatifnya seperti **ReLU**.
* **Feature Scaling Concept:** Memahami pentingnya standardisasi data input agar tidak mendominasi kalkulasi linear di dalam neuron.

### Bagian 2: Lapisan Saraf (Neural Network Layers)
* **Multi-Neuron Layer (Hidden Layer):** Mengubah struktur dari satu neuron menjadi satu lapisan penuh berisi beberapa neuron menggunakan operasi matriks 2 dimensi (`np.dot`).
* **Multi-Layer Architecture:** Menghubungkan *input layer*, *hidden layer*, hingga menghasilkan *output layer*.

### Bagian 3: Evaluasi Model (Loss Functions)
* **Mean Squared Error (MSE):** Mengimplementasikan metrik evaluasi untuk menghitung tingkat kesalahan pada tugas regresi.
* **Binary Cross-Entropy (BCE):** Mengimplementasikan metrik *log loss* berskala penalti ekstrim untuk tugas klasifikasi biner.

### Bagian 4: Pelatihan & Optimasi (Rencana Mendatang)
* **Backpropagation & Gradient Descent:** Algoritma kalkulus untuk memperbarui nilai *weights* dan *bias* secara otomatis agar neuron menjadi semakin cerdas.

---

## Cara Menjalankan Eksperimen

1. Clone repositori ini ke komputer lokal kamu:
   ```bash
   git clone [https://github.com/Npppss/neuron-to-network.git](https://github.com/Npppss/neuron-to-network.git)