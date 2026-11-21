# Sionna OFDM 链路仿真与信道估计对比

基于 NVIDIA Sionna + TensorFlow 实现的 OFDM 物理层链路仿真，包含两组实验：

1. **调制方式对比**：QPSK / 16QAM / 64QAM 在 AWGN 信道下的误码性能
2. **信道估计对比**：瑞利块衰落信道下，LS 与 MMSE 两种信道估计算法的性能差异，并以「理想信道」作为性能上界

项目目的：验证通信原理中的调制阶数与误码率关系、信道估计的噪声抑制机制，并量化 MMSE 相对 LS 的增益来源。

---

## 结果

### 实验一：三种调制方式的误码性能（AWGN）

![调制方式对比](results/ber_curves.png)

| 调制方式 | 每符号比特 | 12 dB 下 BER |
|---|---|---|
| QPSK | 2 | \(1.6\times10^{-5}\) |
| 16QAM | 4 | \(2.9\times10^{-2}\) |
| 64QAM | 6 | \(1.1\times10^{-1}\) |

**结论**：相同信噪比下，调制阶数越高，误码率越差，三者相差约四个数量级。这是速率与可靠性的直接权衡，也是 5G 链路自适应（根据信道质量动态切换调制阶数）的依据。
注意 64QAM 在 0~12 dB 区间只呈现出平缓段，其瀑布下降区在 20~25 dB 附近。

### 实验二：LS 与 MMSE 信道估计对比（瑞利块衰落）

![信道估计对比](results/ber_vs_estimation.png)

| SNR | 理想信道 BER | LS BER | MMSE BER | LS 的 NMSE | MMSE 的 NMSE |
|---|---|---|---|---|---|
| 0 dB | 0.2132 | 0.3136 | 0.2345 | 0.9975 | 0.1131 |
| 5 dB | 0.1043 | 0.1764 | 0.1152 | 0.3169 | 0.0375 |
| 10 dB | 0.0419 | 0.0784 | 0.0469 | 0.0994 | 0.0123 |
| 15 dB | 0.0137 | 0.0272 | 0.0153 | 0.0315 | 0.0041 |
| 20 dB | 0.0050 | 0.0095 | 0.0058 | 0.0101 | 0.0013 |

> 上表为单次仿真结果。由于比特与信道均为随机生成，每次运行会有小幅波动，精确数值以 `results/estimation_results.csv` 为准。

**结论**：

- LS 的估计误差等于噪声本身：0 dB 时 NMSE ≈ 1（0.9975），10 dB 时 NMSE ≈ 0.1（0.0994），与 \(1/\mathrm{SNR}\) 的理论关系一致。
- MMSE 利用「频域平坦」这一先验，把 8 个导频的估计做维纳合并，噪声方差降低至 \(1/N_p\)：0 dB 时 NMSE 为 0.1131，理论值 \(1/8=0.125\)；10 dB 时为 0.0123，理论值 \(1/80=0.0125\)。
- 换算为 dB，MMSE 相对 LS 改善 **9.4 dB**，与 \(10\log_{10}(N_p)=9\ \mathrm{dB}\) 吻合。
- 在误码率上，MMSE 与理想信道仅相差约 0.005（10 dB 时 0.047 对 0.042），而 LS 劣化到 0.078。

---

## 链路结构

```
随机比特
  → 星座映射（QPSK / 16QAM / 64QAM）
  → 资源栅格填装（导频子载波放已知符号，其余放数据）
  → OFDM 调制（IFFT + 加循环前缀 CP）
  → 信道：AWGN 噪声  /  瑞利块衰落 + AWGN
  → OFDM 解调（去 CP + FFT）
  → 信道估计（LS / MMSE，实验二）
  → 均衡（除以信道估计值）
  → 解映射
  → 误码率统计与曲线绘制
```

**关键参数**：子载波数 64，CP 长度 16，调制方式 QPSK（2 bit/符号），导频 8 个 / 64 个子载波（开销 12%）。

---

## 目录结构

```
.
├── code/
│   ├── 01_env_check.py            # 环境自检：跑通一条最小 OFDM 链路
│   ├── 02_ber_curves.py           # 实验一：三种调制方式的 BER 曲线
│   └── 03_channel_estimation.py   # 实验二：LS / MMSE 信道估计对比
├── results/
│   ├── ber_curves.png             # 实验一结果图
│   ├── ber_results.csv            # 实验一原始数据
│   ├── ber_vs_estimation.png      # 实验二结果图
│   └── estimation_results.csv     # 实验二原始数据
├── docs/
│   └── principle.md               # 原理与公式推导说明
├── requirements.txt
└── README.md
```

---

## 环境与运行

环境：Python 3.11 + TensorFlow 2.21.0（CPU）+ Sionna 1.2.2

```bash
pip install -r requirements.txt

# 环境自检
python code/01_env_check.py

# 实验一：调制方式对比（约 40 秒）
python code/02_ber_curves.py

# 实验二：信道估计对比（约 30 秒）
python code/03_channel_estimation.py
```

运行后图片与 CSV 会写入 `results/` 目录。

> **版本提醒**：Sionna 1.x 的接口与 0.x 不同。模块路径为 `sionna.phy.ofdm` 而非 `sionna.ofdm`；QPSK 需写作 `Constellation("qam", 2)`；`OFDMDemodulator` 的 `l_min` 必须 ≤ 0；`Demapper` 调用时必须传入噪声功率。按旧版教程编写会直接报错。

---

## 方法说明

### LS 估计

逐个导频子载波直接相除：

\[
\hat{H}_{\mathrm{LS}}[k] = \frac{Y[k]}{X[k]} = H[k] + \frac{N[k]}{X[k]}
\]

噪声项被完整保留，估计误差的方差等于噪声方差，因此 NMSE 随 SNR 线性下降，0 dB 时约为 1。

### MMSE 估计

在「信道频域平坦、且服从 \(\mathcal{CN}(0,1)\)」的先验下，对 \(N_p\) 个导频的 LS 估计做维纳合并：

\[
\hat{H}_{\mathrm{MMSE}} = \frac{\sigma_H^2}{\sigma_H^2 + \sigma_N^2/N_p}\,\bar{H}_{\mathrm{LS}},
\qquad \sigma_H^2 = 1
\]

等效于把噪声方差压低 \(N_p\) 倍，即获得 \(10\log_{10}(N_p)\) dB 的增益。

---

## 局限性与可扩展方向

- 信道采用**块衰落**模型（符号内频域平坦），因此 MMSE 的先验可以成立。若改为频率选择性衰落，需引入信道频域相关矩阵，MMSE 优势会随先验失配而下降。
- 链路未包含信道编码（LDPC / Polar），因此 BER 曲线为未编码性能，低于 5G 实际系统。
- 均衡后的解映射使用等效噪声方差近似，未计入信道估计误差的完整传播。
- 可扩展：加入 LDPC 编码、多天线 MIMO 预编码、时频二维导频图式，或改为 3GPP TDL 多径信道模型。

---

## 参考

- Sionna 官方文档：https://nvlabs.github.io/sionna/
- Sionna 源码仓库：https://github.com/nvlabs/sionna
