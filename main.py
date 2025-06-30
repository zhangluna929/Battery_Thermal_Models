"""
电池热管理数值模型示例
1. 1-D 厚度方向导热（显式有限差分）
2. 2-D 平面导热（显式有限差分 + 均匀发热）
"""

import numpy as np
import matplotlib.pyplot as plt

# 物性参数
rho = 2400        # kg/m3
cp  = 900         # J/(kg·K)
k   = 1.5         # W/(m·K)
alpha = k / (rho * cp)
q_gen = 5000      # W/m3 发热功率密度

# -----------------------------
# 1. 1-D 模型
# -----------------------------
L  = 1e-3   # 厚度 1 mm
Nx = 50
dx = L / Nx
dt = 0.4 * dx**2 / alpha   # 稳定性条件
Nt = 1000

T = np.full(Nx, 298.15)    # 初始 25 °C
hist = []

for step in range(Nt):
    T_new = T.copy()
    for i in range(1, Nx-1):
        T_new[i] = (T[i] + alpha * dt / dx**2 *
                    (T[i+1] - 2*T[i] + T[i-1]) +
                    q_gen * dt / (rho * cp))
    T_new[0]  = 298.15   # 边界固定温度
    T_new[-1] = 298.15
    T = T_new
    if step % 50 == 0:
        hist.append(T.copy())

plt.figure(figsize=(5, 4))
x_mm = np.linspace(0, L, Nx) * 1e3
for idx, profile in enumerate(hist):
    plt.plot(x_mm, profile - 273.15, label=f"t={idx*50*dt:.1f}s")
plt.xlabel("Thickness / mm"); plt.ylabel("Temperature / °C")
plt.title("1-D Thermal Profile"); plt.legend()
plt.tight_layout(); plt.savefig("thermal_1d.png", dpi=300)
plt.show()

# -----------------------------
# 2. 2-D 模型
# -----------------------------
W, H = 0.02, 0.02    # 2 cm × 2 cm
Nx2, Ny2 = 40, 40
hx, hy   = W / Nx2, H / Ny2
dt2 = 0.4 * min(hx, hy)**2 / (4 * alpha)

T2 = np.full((Ny2, Nx2), 298.15)

for _ in range(300):
    Tn = T2.copy()
    T2[1:-1, 1:-1] = (Tn[1:-1, 1:-1] +
        alpha * dt2 * (
            (Tn[2:, 1:-1] - 2*Tn[1:-1, 1:-1] + Tn[:-2, 1:-1]) / hx**2 +
            (Tn[1:-1, 2:]  - 2*Tn[1:-1, 1:-1] + Tn[1:-1, :-2]) / hy**2
        ) + q_gen * dt2 / (rho * cp))
    T2[0, :] = T2[-1, :] = T2[:, 0] = T2[:, -1] = 298.15

plt.figure(figsize=(5, 4))
plt.imshow(T2 - 273.15, origin="lower",
           extent=[0, W*1000, 0, H*1000], cmap="hot")
plt.colorbar(label="Temperature / °C")
plt.xlabel("mm"); plt.ylabel("mm"); plt.title("2-D Temperature Map")
plt.tight_layout(); plt.savefig("thermal_2d.png", dpi=300)
plt.show()
