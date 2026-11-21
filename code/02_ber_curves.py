"""
阶段二：信噪比扫描，画 BER 曲线

对比 QPSK / 16QAM / 64QAM 三种调制方式在 AWGN 信道下的误码性能。
每个信噪比点跑 1000 个 OFDM 符号，统计误码率，最后画成一条曲线。

运行：
    python 02_BER曲线.py
输出：
    终端打印每个 SNR 的 BER 表格
    同目录生成 ber_curves.png（曲线图）和 ber_results.csv（原始数据）
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")           # 用无界面后端，避免弹窗阻塞，直接存图
import matplotlib.pyplot as plt

from sionna.phy.ofdm import OFDMModulator, OFDMDemodulator
from sionna.phy.mapping import Constellation, Mapper, Demapper
from sionna.phy.channel import AWGN
from sionna.phy.utils import compute_ber

# ---------- 输出目录（结果统一存到仓库的 results/ 下） ----------
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)

# ---------- 链路参数 ----------
fft_size = 64          # 子载波数
cp_len = 16            # 循环前缀长度
num_symbols = 1000     # 每个 SNR 点仿真的 OFDM 符号数（越大 BER 越准，也越慢）
snr_list = [0, 2, 4, 6, 8, 10, 12]
mods = {"QPSK": 2, "16QAM": 4, "64QAM": 6}   # 值 = 每符号比特数

# 调制/解调与 OFDM 模块对所有调制方式通用，噪声也要复用
mod_ = OFDMModulator(cyclic_prefix_length=cp_len)
demod_ = OFDMDemodulator(fft_size=fft_size, l_min=0, cyclic_prefix_length=cp_len)
awgn = AWGN()

results = {}

for name, nbps in mods.items():
    const = Constellation("qam", nbps, normalize=True)   # 2bit 的 qam 就是 QPSK
    mapper = Mapper(constellation=const)
    demapper = Demapper("app", constellation=const, hard_out=True)

    ber_list = []
    for snr_db in snr_list:
        # 发射：随机比特 -> 星座映射 -> OFDM 调制（加 CP）
        bits = tf.random.uniform([num_symbols, fft_size * nbps], 0, 2, tf.int32)
        x = mapper(bits)                 # (符号数, 子载波数)
        tx = mod_(x)

        # 信道：噪声功率 = 1 / 线性信噪比（星座已归一化，符号能量为 1）
        no = 1.0 / (10 ** (snr_db / 10.0))
        rx = awgn(tx, no)

        # 接收：去 CP + FFT -> 解映射 -> 统计误码
        y = demod_(rx)
        bhat = demapper(y, no)
        ber = float(compute_ber(bits, bhat).numpy())

        ber_list.append(ber)
        print(f"{name:>6} | SNR {snr_db:>3} dB | BER = {ber:.6f}")

    results[name] = ber_list
    print("-" * 40)

# ---------- 画图 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(8, 6))
markers = {"QPSK": "o", "16QAM": "s", "64QAM": "^"}
for name, ber_list in results.items():
    plt.semilogy(snr_list, ber_list, marker=markers[name], label=name)

plt.xlabel("信噪比 SNR (dB)")
plt.ylabel("误码率 BER")
plt.title("三种调制方式在 AWGN 信道下的误码性能")
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "ber_curves.png"), dpi=150)

# ---------- 存原始数据，方便以后改图/写报告 ----------
with open(os.path.join(OUT, "ber_results.csv"), "w", encoding="utf-8") as f:
    f.write("SNR_dB," + ",".join(results.keys()) + "\n")
    for i, snr in enumerate(snr_list):
        f.write(str(snr) + "," + ",".join(f"{results[m][i]:.8f}" for m in results) + "\n")

print("已保存 results/ber_curves.png 和 results/ber_results.csv")
