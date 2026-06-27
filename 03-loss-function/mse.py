import numpy as np

def hitung_mse(jawaban_asli,tebakan_neuron):
    ## menghoitung selisih mean squared error (mse)
    selisih=jawaban_asli - tebakan_neuron
    #mengkuadratkan selisih pangkat 2
    selisih_kuadrat=selisih**2
    #menghitung rata-rata dari selisih kuadrat
    nilai_loss=np.mean(selisih_kuadrat)

    return nilai_loss


jawaban_asli=np.array([1.5, 2.0, 3.2])
tebakan_neuron=np.array([5.0, 2.5, 3.8])
loss_saat_ini=hitung_mse(jawaban_asli,tebakan_neuron)
print(f"Loss saat ini adalah: {loss_saat_ini:.4f}")
