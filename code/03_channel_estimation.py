"""
阶段三：信道估计 —— LS 与 MMSE 对比

场景：瑞利块衰落信道（每个 OFDM 符号经历一个独立的复高斯衰落，频域上平坦）
导频：梳状导频（每个 OFDM 符号里固定放 8 个子载波做导频，其余 56 个传数据）

对比三种接收方式：
  1. 理想信道（Perfect CSI）—— 假装接收端已经知道真实信道，性能上界
  2. LS 估计 —— 每个导频子载波独立相除，不做任何降噪
  3. MMSE 估计 —— 利用"频域平坦"这个先验，把 8 个导频的估计做加权合并来抑制噪声

输出：ber_vs_estimation.png（BER 和 NMSE 两张子图）、estimation_results.csv

运行：python 03_信道估计对比.py
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sionna.phy.ofdm import OFDMModulator, OFDMDemodulator
from sionna.phy.mapping import Constellation, Mapper, Demapper
from sionna.phy.utils import compute_ber

# ---------------- 输出目录（结果统一存到仓库的 results/ 下） ----------------
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)

# ---------------- 参数 ----------------
fft_size = 64
cp_len = 16
nbps = 2                     # QPSK
num_sym = 50                 # 每帧的 OFDM 符号数
num_frames = 40              # 每个 SNR 点重复多少帧
snr_list = [0, 5, 10, 15, 20]

# 导频布置：每 8 个子载波放一个导频，共 8 个
pilot_idx = np.arange(0, fft_size, 8)
data_idx = np.setdiff1d(np.arange(fft_size), pilot_idx)
num_pilots = len(pilot_idx)
num_data_sc = len(data_idx)

# 每个数据子载波用最近的导频做 LS 插值（频域平坦，所以最近邻就够）
nearest_pilot = np.array([np.argmin(np.abs(pilot_idx - k)) for k in data_idx])

# ---------------- 模块 ----------------
const = Constellation("qam", nbps, normalize=True)
mapper = Mapper(constellation=const)
demapper = Demapper("app", constellation=const, hard_out=True)
mod = OFDMModulator(cyclic_prefix_length=cp_len)
demod = OFDMDemodulator(fft_size=fft_size, l_min=0, cyclic_prefix_length=cp_len)

print(f"导频子载波 {num_pilots} 个，数据子载波 {num_data_sc} 个，导频开销 {num_pilots/fft_size:.0%}")

results = {"理想信道": {"ber": [], "nmse": []},
           "LS":      {"ber": [], "nmse": []},
           "MMSE":    {"ber": [], "nmse": []}}


def build_frame():
    """生成一帧：随机比特 -> QPSK -> 填进资源栅格（导频位置填 1）"""
    bits = tf.random.uniform([num_sym, num_data_sc * nbps], 0, 2, tf.int32)
    data_sym = mapper(bits).numpy()                 # (符号数, 数据子载波)

    grid = np.zeros((num_sym, fft_size), np.complex64)
    grid[:, pilot_idx] = 1.0 + 0.0j                 # 导频符号事先约定为 1
    grid[:, data_idx] = data_sym
    return bits, tf.constant(grid), data_sym


for snr_db in snr_list:
    no = 1.0 / (10 ** (snr_db / 10.0))              # 噪声功率，信号功率归一化为 1

    acc = {k: {"err": 0.0, "nmse": 0.0, "n": 0} for k in results}

    for _ in range(num_frames):
        bits, grid, data_sym = build_frame()
        tx = mod(grid)                              # OFDM 调制（加 CP），输出是一维时域序列
        tx = tf.reshape(tx, [num_sym, fft_size + cp_len])   # 还原成 (符号数, 每符号采样数)

        # 瑞利块衰落：每个 OFDM 符号一个复增益，CN(0,1)
        h = tf.complex(tf.random.normal([num_sym, 1]) / np.sqrt(2.0),
                       tf.random.normal([num_sym, 1]) / np.sqrt(2.0))
        rx = h * tx

        # 加复高斯白噪声：实部虚部各 no/2 方差
        noise = tf.complex(tf.random.normal(tf.shape(rx)) * np.sqrt(no / 2.0),
                           tf.random.normal(tf.shape(rx)) * np.sqrt(no / 2.0))
        rx = tf.reshape(rx + noise, [-1])
        y = demod(rx)                               # 去 CP + FFT

        h_true = h                                  # 真实信道（每符号一个标量）

        # ---- LS：每个导频子载波各自相除，噪声原样保留 ----
        y_pil = tf.gather(y, pilot_idx, axis=1)
        x_pil = tf.gather(grid, pilot_idx, axis=1)
        h_ls_pil = y_pil / x_pil                            # (符号数, 8)
        h_ls = tf.gather(h_ls_pil, nearest_pilot, axis=1)   # 插值到数据子载波

        # ---- MMSE：频域平坦是已知先验，把 8 个导频做维纳加权合并 ----
        h_ls_mean = tf.reduce_mean(h_ls_pil, axis=1, keepdims=True)   # (符号数, 1)
        # 维纳收缩系数 = σ_h² / (σ_h² + 噪声方差/导频数)，σ_h² = 1
        w = 1.0 / (1.0 + no / num_pilots)
        h_mmse = w * h_ls_mean
        h_mmse = tf.repeat(h_mmse, num_data_sc, axis=1)

        for name, h_hat in [("理想信道", h_true), ("LS", h_ls), ("MMSE", h_mmse)]:
            # 均衡：数据子载波除以信道估计
            y_data = tf.gather(y, data_idx, axis=1)
            x_est = y_data / h_hat

            # 均衡后的等效噪声方差 = 原噪声 / |h_hat|^2
            no_eff = no / tf.math.square(tf.abs(h_hat))

            bhat = demapper(x_est, no_eff)
            ber = float(compute_ber(bits, bhat).numpy())

            # 归一化均方误差：衡量信道估得准不准
            nmse = float(tf.reduce_mean(tf.square(tf.abs(h_hat - h_true))).numpy())
            if name == "理想信道":
                nmse = 0.0

            acc[name]["err"] += ber * int(tf.size(bits))
            acc[name]["nmse"] += nmse
            acc[name]["n"] += 1

    for name in results:
        results[name]["ber"].append(acc[name]["err"] / (acc[name]["n"] * int(tf.size(bits))))
        results[name]["nmse"].append(acc[name]["nmse"] / acc[name]["n"])

    print(f"SNR {snr_db:>2} dB | BER 理想 {results['理想信道']['ber'][-1]:.5f} | "
          f"LS {results['LS']['ber'][-1]:.5f} | MMSE {results['MMSE']['ber'][-1]:.5f} || "
          f"NMSE LS {results['LS']['nmse'][-1]:.4f} | MMSE {results['MMSE']['nmse'][-1]:.4f}")

# ---------------- 画图 ----------------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
styles = {"理想信道": ("k", "o"), "LS": ("tab:blue", "s"), "MMSE": ("tab:red", "^")}

# 左图：误码率
for name in results:
    c, m = styles[name]
    axes[0].semilogy(snr_list, results[name]["ber"], marker=m, color=c, label=name)
axes[0].set_xlabel("信噪比 SNR (dB)")
axes[0].set_ylabel("误码率 BER")
axes[0].set_title("信道估计对误码率的影响")
axes[0].grid(True, which="both", ls="--", alpha=0.6)
axes[0].legend()

# 右图：信道估计误差
for name in ["LS", "MMSE"]:
    c, m = styles[name]
    axes[1].semilogy(snr_list, results[name]["nmse"], marker=m, color=c, label=name)
axes[1].set_xlabel("信噪比 SNR (dB)")
axes[1].set_ylabel("信道估计归一化均方误差 NMSE")
axes[1].set_title("估计精度：LS vs MMSE")
axes[1].grid(True, which="both", ls="--", alpha=0.6)
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT, "ber_vs_estimation.png"), dpi=150)

with open(os.path.join(OUT, "estimation_results.csv"), "w", encoding="utf-8") as f:
    f.write("SNR_dB," + ",".join(f"{k}_BER" for k in results) + "," + ",".join(f"{k}_NMSE" for k in results) + "\n")
    for i, snr in enumerate(snr_list):
        f.write(f"{snr}," + ",".join(f"{results[k]['ber'][i]:.8f}" for k in results) + ","
                + ",".join(f"{results[k]['nmse'][i]:.8f}" for k in results) + "\n")

print("已保存 results/ber_vs_estimation.png 和 results/estimation_results.csv")
