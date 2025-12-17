import math
import time
from typing import List

PI = math.pi

def generate_s_nt(n: int) -> List[float]:
    value = 96 + n
    bin_str = bin(value)[2:].zfill(8)[-8:]
    return [float(b) for b in bin_str]

class FourierCoefficient:
    def __init__(self, ak=0.0, bk=0.0):
        self.Ak = ak
        self.Bk = bk

class OperationCount:
    def __init__(self):
        self.additions = 0
        self.multiplications = 0

def calculate_dft_term(fi, k, i, N):
    angle = 2.0 * PI * k * i / N
    return fi * math.cos(angle), -fi * math.sin(angle)

def calculate_fourier_coefficient(signal, k, N, ops: OperationCount):
    sum_Ak = 0.0
    sum_Bk = 0.0

    for i in range(N):
        real, imag = calculate_dft_term(signal[i], k, i, N)
        sum_Ak += real
        sum_Bk += imag

        ops.multiplications += 2
        ops.additions += 2

    ops.multiplications += 2
    return FourierCoefficient(sum_Ak / N, sum_Bk / N)

def bit_reverse(x, log2n):
    n = 0
    for _ in range(log2n):
        n = (n << 1) | (x & 1)
        x >>= 1
    return n

def fft(a, invert, ops: OperationCount):
    n = len(a)
    log2n = n.bit_length() - 1

    for i in range(n):
        j = bit_reverse(i, log2n)
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    while length <= n:
        ang = 2 * PI / length * (-1 if invert else 1)
        wlen = complex(math.cos(ang), math.sin(ang))

        for i in range(0, n, length):
            w = 1 + 0j
            for j in range(length // 2):
                u = a[i + j]
                v = a[i + j + length // 2] * w

                a[i + j] = u + v
                a[i + j + length // 2] = u - v

                ops.multiplications += 8
                ops.additions += 8

                w *= wlen
                ops.multiplications += 4
                ops.additions += 2

        length <<= 1

    if not invert:
        for i in range(n):
            a[i] /= n
            ops.multiplications += 2

def main():

    n_variant = 20
    base_signal = generate_s_nt(n_variant)
    N = 8

    print(f"Number N = 96 + {n_variant} = {96 + n_variant}")
    print("Vector s(nT) (8 samples):")
    print(base_signal)

    dft_ops = OperationCount()
    dft_results = []

    t0 = time.perf_counter()
    for k in range(N):
        dft_results.append(
            calculate_fourier_coefficient(base_signal, k, N, dft_ops)
        )
    t1 = time.perf_counter()
    dft_time = (t1 - t0) * 1e6

    fft_ops = OperationCount()
    fft_signal = [complex(v, 0) for v in base_signal]

    t0 = time.perf_counter()
    fft(fft_signal, False, fft_ops)
    t1 = time.perf_counter()
    fft_time = (t1 - t0) * 1e6

    print("\n1. Results Comparison")
    print(f"{'k':<5}{'DFT':<30}{'FFT'}")

    for k in range(N):
        dft = dft_results[k]
        fftc = fft_signal[k]
        print(f"{k:<5}"
              f"{dft.Ak:+.3f} {'+ j' if dft.Bk >= 0 else '- j'}{abs(dft.Bk):.3f}"
              f"{'':<10}"
              f"{fftc.real:+.3f} {'+ j' if fftc.imag >= 0 else '- j'}{abs(fftc.imag):.3f}")

    print("\n2. Performance Comparison")
    print(f"{'Metric':<25}{'DFT':<20}{'FFT'}")
    print(f"{'Time (microseconds)':<25}{dft_time:<20.1f}{fft_time:.1f}")
    print(f"{'Multiplications':<25}{dft_ops.multiplications:<20}{fft_ops.multiplications}")
    print(f"{'Additions':<25}{dft_ops.additions:<20}{fft_ops.additions}")

if __name__ == "__main__":
    main()
