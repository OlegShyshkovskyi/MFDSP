import numpy as np
import matplotlib.pyplot as plt
import math
PI = math.pi

def f(x, n):
    x_arr = np.array(x, dtype=float)
    return (x_arr ** n) * np.exp(-(x_arr**2) / n)
def _integrate(func, a, b, args=(), Ntrapz=50001):
        xs = np.linspace(a, b, Ntrapz)
        ys = func(xs, *args)
        return np.trapz(ys, xs)
def a_k(k, n):
    res = _integrate(lambda x, n_: f(x, n_) * np.cos(k * x), -PI, PI, args=(n,))
    return res / PI

def b_k(k, n):
    if k == 0:
        return 0.0
    res = _integrate(lambda x, n_: f(x, n_) * np.sin(k * x), -PI, PI, args=(n,))
    return res / PI

def compute_coeffs(N, n):
    ak = [0.0] * (N + 1)
    bk = [0.0] * (N + 1)
    for k in range(0, N + 1):
        ak[k] = a_k(k, n)
        if k >= 1:
            bk[k] = b_k(k, n)
    bk[0] = 0.0
    return ak, bk
def fourier_series(x, N, n, coeffs=None):
    x = np.array(x, dtype=float)
    if coeffs is None:
        coeffs = compute_coeffs(N, n)
    ak, bk = coeffs
    s = np.full_like(x, ak[0] / 2.0, dtype=float)
    for k in range(1, N + 1):
        s += ak[k] * np.cos(k * x) + bk[k] * np.sin(k * x)
    return s

def plot_time_domain(n, N, coeffs=None, show=True):
    xs = np.linspace(-PI, PI, 2000)
    ys = f(xs, n)
    print("Точне аналітичне обчислення значення функції:")
    print(ys)
    ys_approx = fourier_series(xs, N, n, coeffs=coeffs)
    plt.figure(figsize=(10,5))
    plt.plot(xs, ys, label=f'f(x), n={n}', linewidth=2)
    plt.plot(xs, ys_approx, '--', label=f'Fourier S_N, N={N}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Функція та наближення рядом Фур’є')
    plt.legend()
    plt.grid(True)
    if show:
        plt.show()

def plot_frequency_domain(ak, bk, show=True):
    ks = np.arange(len(ak))
    amp = np.sqrt(np.array(ak)**2 + np.array(bk)**2)

    plt.figure(figsize=(10,4))
    plt.stem(ks, ak, basefmt=" ", linefmt='C0-', markerfmt='C0o', label='a_k')
    plt.stem(ks, bk, basefmt=" ", linefmt='C1-', markerfmt='C1x', label='b_k')
    plt.xlabel('k')
    plt.ylabel('coef value')
    plt.title('Коефіцієнти a_k та b_k')
    plt.legend()
    plt.grid(True)
    if show:
        plt.show()

    plt.figure(figsize=(8,4))
    plt.stem(ks, amp, basefmt=" ")
    plt.xlabel('k')
    plt.ylabel('amplitude')
    plt.title('Амплітуда гармонік A_k = sqrt(a_k^2 + b_k^2)')
    plt.grid(True)
    if show:
        plt.show()

def relative_error(n, N, coeffs=None, sample_points=2000):
    xs = np.linspace(-PI, PI, sample_points)
    ys = f(xs, n)
    ys_approx = fourier_series(xs, N, n, coeffs=coeffs)
    mse = np.mean((ys - ys_approx)**2)
    denom = np.mean(ys**2)
    rel_rms = math.sqrt(mse / denom) if denom != 0 else float('nan')
    rmse = math.sqrt(mse)
    return {'rmse': rmse, 'rel_rms': rel_rms}

def save_results(filename, N, ak, bk, error_dict):
    with open(filename, 'w', encoding='utf-8') as fout:
        fout.write(f"Fourier approximation results\n")
        fout.write(f"Order N = {N}\n\n")
        fout.write("k, a_k, b_k\n")
        for k in range(len(ak)):
            fout.write(f"{k}, {ak[k]:.12e}, {bk[k]:.12e}\n")
        fout.write("\nErrors:\n")
        fout.write(f"RMSE = {error_dict['rmse']:.12e}\n")
        fout.write(f"Relative RMS error = {error_dict['rel_rms']:.12e}\n")
    print(f"Results saved to {filename}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fourier approximation (even student).")
    parser.add_argument('--n', type=int, default=10,
                        help='student number n (even variant). Default 10.')
    parser.add_argument('--N', type=int, default=10,
                        help='Fourier order N (partial sum). Default 10.')
    parser.add_argument('--outfile', type=str, default='results.txt',
                        help='output filename for coefficients and error')
    parser.add_argument('--no-plots', action='store_true', help='do not display plots')
    args = parser.parse_args()

    n = args.n
    N = args.N

    print(f"Running Fourier approximation for n={n}, N={N} ...")
    ak, bk = compute_coeffs(N, n)
    errs = relative_error(n, N, coeffs=(ak,bk))
    save_results(args.outfile, N, ak, bk, errs)

    if not args.no_plots:
        plot_time_domain(n, N, coeffs=(ak,bk))
        plot_frequency_domain(ak, bk)

    print("Done.")
    print(f"Relative RMS error = {errs['rel_rms']:.6e}, RMSE = {errs['rmse']:.6e}")
