import numpy as np
import matplotlib.pyplot as plt

PI = np.pi
n = 20
T_values = [4, 8, 16, 32, 64, 128]
N = 100 * n
dt = 0.1
max_k = 20

def f(t):
    return np.sign(t) * (abs(t) ** ((2 * n + 1) / 3))

def calculate_fourier_member(k, T):
    wk = 2 * PI * k / T
    re_sum = 0.0
    im_sum = 0.0

    t = -N
    while t <= N:
        ft = f(t)
        arg = -wk * t
        re_sum += ft * np.cos(arg) * dt
        im_sum += ft * np.sin(arg) * dt
        t += dt

    return re_sum, im_sum

def amplitude_spectrum(Re, Im):
    return np.sqrt(Re**2 + Im**2)

print(f"Test calculation for n = {n}")
k_test = 1
T_test = 12.0

Re_test, Im_test = calculate_fourier_member(k_test, T_test)
Amp_test = amplitude_spectrum(Re_test, Im_test)

print(f"k = {k_test}, T = {T_test}")
print(f"Re F(w_k) = {Re_test}")
print(f"Im F(w_k) = {Im_test}")
print(f"|F(w_k)|  = {Amp_test}")

for T in T_values:
    print(f"\n========== Period T = {T} ==========")
    print("k\tRe F(wk)\tIm F(wk)\t|F(wk)|")

    k_values = []
    Re_values = []
    Amp_values = []

    for k in range(0, max_k + 1):
        Re, Im = calculate_fourier_member(k, T)
        Amp = amplitude_spectrum(Re, Im)

        print(f"{k}\t{Re:.6e}\t{Im:.6e}\t{Amp:.6e}")

        k_values.append(k)
        Re_values.append(Re)
        Amp_values.append(Amp)

    plt.figure(figsize=(10, 5))
    plt.plot(k_values, Re_values, 'bo-', label='Re F(wk)')
    plt.plot(k_values, Amp_values, 'ro-', label='|F(wk)|')

    plt.title(f'Spectrum for T = {T}')
    plt.xlabel('k (Harmonic number)')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
