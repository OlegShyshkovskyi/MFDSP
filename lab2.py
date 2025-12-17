import argparse
import math
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def generate_binary_vector(n: int) -> List[float]:
    val = 96 + n
    bin_str = bin(val)[2:]
    
    prefix = '0' if (n % 2 == 0) else '1'
    
    full_str = prefix + bin_str

    if len(full_str) < 8:
        full_str = full_str.zfill(8)
    elif len(full_str) > 8:
        full_str = full_str[-8:]
        
    return [float(bit) for bit in full_str]

def dft_term_trig(x: List[float], k: int, N: int) -> complex:
    re = 0.0
    im = 0.0
    for n, xn in enumerate(x):
        angle = 2.0 * math.pi * k * n / N
        c = math.cos(angle)
        s = math.sin(angle)
        re += xn * c
        im += - xn * s
    return complex(re, im)

def dft_full(x: List[float]) -> List[complex]:
    N = len(x)
    X = [0j] * N
    for k in range(N):
        X[k] = dft_term_trig(x, k, N)
    return X

def idft_full(X: List[complex]) -> List[complex]:
    N = len(X)
    x = [0j] * N
    for n in range(N):
        acc = 0+0j
        for k in range(N):
            angle = 2.0 * math.pi * k * n / N
            acc += X[k] * complex(math.cos(angle), math.sin(angle))
        x[n] = acc / N
    return x

def amplitude_phase(X: List[complex]) -> Tuple[List[float], List[float]]:
    amps = [abs(v) for v in X]
    phases = [math.atan2(v.imag, v.real) for v in X]
    return amps, phases

def dft_ops_count(N: int) -> Tuple[int,int]:
    mults = 2 * N * N
    adds = 2 * N * (N - 1)
    return mults, adds

def generate_input_vector(N: int, mode: str='synthetic', seed: int=None) -> List[float]:
    if seed is not None:
        np.random.seed(seed)
    if mode == 'random':
        return list(np.random.randn(N))
    t = np.arange(N)
    f1 = 1.0
    f2 = 3.0
    x = 1.5 * np.sin(2*np.pi * f1 * t / N) + 0.8 * np.cos(2*np.pi * f2 * t / N)
    x += 0.05 * np.random.randn(N)
    return list(x)

def reconstruct_continuous(X: List[complex], T: float=1.0, points: int=1000) -> Tuple[np.ndarray, np.ndarray]:
    N = len(X)
    t = np.linspace(0, T, points, endpoint=False)
    s = np.zeros_like(t, dtype=complex)
    for k, Xk in enumerate(X):
        s += Xk * np.exp(1j * 2 * np.pi * k * t / T)
    s = s / N
    return t, s.real

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1, help='student number n (used to compute N = 10 + n)')
    p.add_argument('--rand', action='store_true', help='generate random input vector')
    p.add_argument('--seed', type=int, default=None, help='seed for random generation')
    p.add_argument('--load', type=str, default=None, help='load CSV of samples (one column)')
    args = p.parse_args()

    n = 20
    N = 10 + n
    print(f"n = {n}, so N = 10 + n = {N}")
    if args.load:
        xs = []
        with open(args.load, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                xs.append(float(row[0]))
        if len(xs) != N:
            print(f"Warning: loaded {len(xs)} samples but expected N={N}; resizing/truncating/padding zeros.")
            if len(xs) > N:
                xs = xs[:N]
            else:
                xs = xs + [0.0]*(N - len(xs))
    else:
        mode = 'random' if args.rand else 'synthetic'
        xs = generate_input_vector(N, mode=mode, seed=args.seed)

    print("Input x[n]:")
    print(np.array(xs))

    k_example = 1 if N > 1 else 0
    t0 = time.perf_counter()
    Xk = dft_term_trig(xs, k_example, N)
    t1 = time.perf_counter()
    print(f"\nExample DFT term k={k_example}: X[{k_example}] = {Xk.real:.6f} + j{Xk.imag:.6f}")
    print(f"Time for single k-term (trig form): {(t1 - t0)*1000:.3f} ms")

    t_start = time.perf_counter()
    X = dft_full(xs)
    t_end = time.perf_counter()
    elapsed_ms = (t_end - t_start) * 1000.0
    print(f"\nComputed full DFT (naive O(N^2)) in {elapsed_ms:.3f} ms")

    mults, adds = dft_ops_count(N)
    print(f"Estimated arithmetic operations (naive model): multiplications ≈ {mults}, additions ≈ {adds}")
    print("Note: trig function costs not included in these counts.")

    amps, phases = amplitude_phase(X)

    print("\nFirst coefficients Ck (complex):")
    for k in range(min(8, N)):
        print(f"k={k:2d}: {X[k].real:.6f} + j{X[k].imag:.6f} |A|={amps[k]:.6f} arg={phases[k]:.6f}")

    ks = np.arange(N)
    plt.figure(figsize=(10,4))
    plt.stem(ks, amps)
    plt.title("Amplitude spectrum |C_k|")
    plt.xlabel("k")
    plt.ylabel("|C_k|")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10,4))
    phases_unwrapped = np.unwrap(phases)
    plt.stem(ks, phases_unwrapped)
    plt.title("Phase spectrum arg(C_k)")
    plt.xlabel("k")
    plt.ylabel("phase (radians)")
    plt.grid(True)
    plt.show()

    x_rec = idft_full(X)
    print("\nReconstruction error (max abs) between original and IDFT-reconstructed samples:")
    recon_err = max(abs(x_rec[n].real - xs[n]) for n in range(N))
    print(f"max abs error = {recon_err:.6e}")

    T = 1.0
    t_cont, s_cont = reconstruct_continuous(X, T=T, points=2000)

    plt.figure(figsize=(10,4))
    plt.plot(t_cont, s_cont, label='Reconstructed s(t)')
    sample_times = np.arange(N) * (T / N)
    plt.plot(sample_times, xs, 'o', label='samples x[n]')
    plt.title("Analog-like reconstruction via inverse DFT")
    plt.xlabel("t (normalized)")
    plt.ylabel("s(t)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nInverse DFT sample values s[n] (showing first few):")
    for n in range(min(6, N)):
        val = x_rec[n]
        print(f"n={n}: s[{n}] = {val.real:.6f} + j{val.imag:.6f}")

    my_binary_vector = generate_binary_vector(20)
    print(f"\nNumber N = 96 + 20 = 116")
    print(f"Vector s(nT) (8 samples):")
    print(my_binary_vector)

if __name__ == "__main__":
    main()