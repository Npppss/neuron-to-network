import numpy as np

def hitung_bce(jawaban_asli, tebakan_neuron):
    # Episilon untuk menghindari log(0)
    # karena log(0) tidak terdefinisi, kita menambahkan nilai kecil (epsilon) ke tebakan_neuron 
    epsilon = 1e-15
    #kita "jepit" nilai tebakan_neuron agar tetap berada dalam rentang 0-1
    tebakan_aman=np.clip(tebakan_neuron, epsilon, 1 - epsilon)
    #rumus binary cross entropy (BCE)
    loss_tiap_data = - (jawaban_asli * np.log(tebakan_aman) + (1 - jawaban_asli) * np.log(1 - tebakan_aman))
    #menghitung rata-rata dari loss tiap data
    nilai_loss = np.mean(loss_tiap_data)

    return nilai_loss

jawaban_asli = np.array([1, 0, 1])
tebakan_neuron = np.array([0.9, 0.1, 0.8])
loss_saat_ini = hitung_bce(jawaban_asli, tebakan_neuron)

print(f"Loss saat ini adalah: {loss_saat_ini:.4f}")