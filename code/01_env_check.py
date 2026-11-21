"""
Sionna 环境验证脚本（第 0 步：确认环境能跑通）

链路：随机比特 -> QPSK 映射 -> OFDM 调制(加CP) -> AWGN 信道 -> OFDM 解调(去CP) -> 解映射 -> 统计 BER

在 sionna-env 环境里运行：
    python 01_环境验证.py
"""

import tensorflow as tf
from sionna.phy.ofdm import OFDMModulator, OFDMDemodulator
from sionna.phy.mapping import Constellation, Mapper, Demapper
from sionna.phy.channel import AWGN
from sionna.phy.utils import compute_ber

# ---------- 参数 ----------
fft_size = 64        # 子载波数（FFT 点数）
cp_len = 16          # 循环前缀长度（抗多径）
nbps = 2             # 每符号比特数：2 -> QPSK
num_symbols = 100    # OFDM 符号数
snr_db = 10.0        # 信噪比 (dB)

# ---------- 模块 ----------
# Sionna 1.x 里 QPSK 用 constellation_type="qam" + num_bits_per_symbol=2 表示
const = Constellation("qam", nbps, normalize=True)
mapper = Mapper(constellation=const)
demapper = Demapper("app", constellation=const, hard_out=True)
mod = OFDMModulator(cyclic_prefix_length=cp_len)
demod = OFDMDemodulator(fft_size=fft_size, l_min=0, cyclic_prefix_length=cp_len)
awgn = AWGN()

# ---------- 发射 ----------
bits = tf.random.uniform([num_symbols, fft_size * nbps], 0, 2, tf.int32)
x = mapper(bits)                       # (符号数, 子载波数)
tx = mod(x)                            # 加 CP，展平成时域采样
print("OFDM 调制后长度:", tx.shape)

# ---------- 信道 ----------
no = 1.0 / (10 ** (snr_db / 10.0))     # 噪声功率（线性值）
rx = awgn(tx, no)                      # 加高斯白噪声

# ---------- 接收 ----------
y = demod(rx)                          # 去 CP + FFT
bhat = demapper(y, no)                 # 解映射（硬判决），需传入噪声功率
ber = compute_ber(bits, bhat)

print("SNR =", snr_db, "dB")
print("BER =", float(ber.numpy()))
print("环境正常，可以开始做项目了")
