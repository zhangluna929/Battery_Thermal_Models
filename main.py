"""
Battery Thermal Models — Detailed Version (1‑D & 2‑D Explicit FDM)

"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Tuple, Dict
import time


np.set_printoptions(precision=3, suppress=True)
plt.style.use("seaborn-v0_8-whitegrid")


# 参数与默认物性（以 18650 电芯为例）

DEFAULT_MATERIAL: Dict[str, float] = {
    "rho": 2400,   # kg/m^3
    "cp":   900,   # J/(kg·K)
    "k":    1.5,   # W/(m·K)
}

# -----------------------------------------------------------------------------
# 边界条件类型枚举
# -----------------------------------------------------------------------------
class BCType:
    ADIABATIC = "adiabatic"
    FIXED     = "fixed"
    CONVECT   = "convective"



class Solver1D:
    """显式差分 1‑D 求解器"""

    def __init__(
        self,
        thickness: float,
        nx: int,
        dt: float,
        t_total: float,
        material: Dict[str, float] = DEFAULT_MATERIAL,
        q_dot: float | Callable[[float], float] = 0.0,
        bc_left: Tuple[str, float] = (BCType.FIXED, 298.15),
        bc_right: Tuple[str, float] = (BCType.FIXED, 298.15),
        init_func: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.L = thickness
        self.nx = nx
        self.dx = thickness / nx
        self.dt = dt
        self.steps = int(np.ceil(t_total / dt))
        self.q_dot = q_dot  # W/m3 或函数 t->W/m3
        self.alpha = material["k"] / (material["rho"] * material["cp"])
        self.material = material
        self.bc_left = bc_left
        self.bc_right = bc_right
        self.x = np.linspace(0, thickness, nx)
        self.time_hist: list[np.ndarray] = []

     # 初温
        if init_func is None:
            self.T = np.full(nx, 298.15)
        else:
            self.T = init_func(self.x)


        Fo = self.alpha * dt / self.dx ** 2
        if Fo > 0.5:
            raise ValueError(
                f"时间步过大导致不稳定：Fo={Fo:.3f}>0.5，应减小 dt 或增大 nx")


    def _apply_boundary(self, T_new: np.ndarray):
        # 左边界
        bc_type, val = self.bc_left
        if bc_type == BCType.FIXED:
            T_new[0] = val
        elif bc_type == BCType.ADIABATIC:
            T_new[0] = T_new[1]
        elif bc_type == BCType.CONVECT:
            h, T_inf = val
            k = self.material["k"]
            T_new[0] = (h * self.dx * T_inf + k * T_new[1]) / (h * self.dx + k)


        bc_type, val = self.bc_right
        if bc_type == BCType.FIXED:
            T_new[-1] = val
        elif bc_type == BCType.ADIABATIC:
            T_new[-1] = T_new[-2]
        elif bc_type == BCType.CONVECT:
            h, T_inf = val
            k = self.material["k"]
            T_new[-1] = (h * self.dx * T_inf + k * T_new[-2]) / (h * self.dx + k)


    def step_time(self, t: float):
        T_new = self.T.copy()
        k, rho, cp = self.material["k"], self.material["rho"], self.material["cp"]
        q_val = self.q_dot(t) if callable(self.q_dot) else self.q_dot
        for i in range(1, self.nx - 1):
            T_new[i] = (
                self.T[i]
                + self.alpha * self.dt / self.dx ** 2 * (self.T[i+1] - 2*self.T[i] + self.T[i-1])
                + q_val * self.dt / (rho * cp)
            )

        self._apply_boundary(T_new)
        self.T[:] = T_new


    def run(self, record_interval: float = 1.0):
        record_steps = max(1, int(record_interval / self.dt))
        for n in range(self.steps):
            current_time = (n + 1) * self.dt
            self.step_time(current_time)
            if n % record_steps == 0:
                self.time_hist.append(self.T.copy())
        return np.array(self.time_hist)


    def plot_profiles(self):
        plt.figure(figsize=(5, 4))
        for idx, prof in enumerate(self.time_hist[:: max(len(self.time_hist)//6, 1) ]):
            label = f"t={idx * len(self.time_hist)//6 * self.dt:.0f}s"
            plt.plot(self.x*1e3, prof-273.15, label=label)
        plt.xlabel("Thickness / mm"); plt.ylabel("Temperature / °C")
        plt.title("1‑D Temperature Profiles")
        plt.legend(); plt.tight_layout()
        Path("results").mkdir(exist_ok=True)
        plt.savefig("results/thermal_1d_profiles.png", dpi=300)
        plt.show()


class Solver2D:
    """显式差分 2‑D 求解器 (XY 平面)"""

    def __init__(
        self,
        width: float,
        height: float,
        nx: int,
        ny: int,
        dt: float,
        t_total: float,
        material: Dict[str, float] = DEFAULT_MATERIAL,
        q_dot: float | Callable[[float], float] = 0.0,
        bc_type: str = BCType.FIXED,
        bc_value: float = 298.15,
        init_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ):
        self.W, self.H = width, height
        self.nx, self.ny = nx, ny
        self.dx, self.dy = width / nx, height / ny
        self.dt = dt
        self.steps = int(np.ceil(t_total / dt))
        self.q_dot = q_dot
        self.alpha = material["k"] / (material["rho"] * material["cp"])
        self.material = material
        self.bc_type = bc_type
        self.bc_value = bc_value


        self.x = np.linspace(0, width, nx)
        self.y = np.linspace(0, height, ny)
        X, Y = np.meshgrid(self.x, self.y)
        if init_func is None:
            self.T = np.full((ny, nx), 298.15)
        else:
            self.T = init_func(X, Y)


        coef = self.alpha * dt * (1/self.dx**2 + 1/self.dy**2)
        if coef > 0.5:
            raise ValueError(f"时间步导致2‑D显式不稳定：coef={coef:.3f}>0.5")

        self.time_hist: list[np.ndarray] = []


    def _apply_boundary(self, T_new: np.ndarray):
        if self.bc_type == BCType.FIXED:
            T_new[0, :] = T_new[-1, :] = T_new[:, 0] = T_new[:, -1] = self.bc_value
        elif self.bc_type == BCType.ADIABATIC:
            T_new[0, :] = T_new[1, :]
            T_new[-1, :] = T_new[-2, :]
            T_new[:, 0] = T_new[:, 1]
            T_new[:, -1] = T_new[:, -2]



    def step_time(self, t: float):
        Tn = self.T.copy()
        T_new = Tn.copy()
        q_val = self.q_dot(t) if callable(self.q_dot) else self.q_dot
        k, rho, cp = self.material["k"], self.material["rho"], self.material["cp"]

        T_new[1:-1,1:-1] = (
            Tn[1:-1,1:-1]
            + self.alpha * self.dt * (
                (Tn[2:,1:-1] - 2*Tn[1:-1,1:-1] + Tn[:-2,1:-1]) / self.dx**2
                + (Tn[1:-1,2:]   - 2*Tn[1:-1,1:-1] + Tn[1:-1,:-2]) / self.dy**2
            )
            + q_val * self.dt / (rho * cp)
        )
        self._apply_boundary(T_new)
        self.T[:] = T_new


    def run(self, record_interval: float = 1.0):
        record_steps = max(1, int(record_interval / self.dt))
        for n in range(self.steps):
            current_time = (n + 1) * self.dt
            self.step_time(current_time)
            if n % record_steps == 0:
                self.time_hist.append(self.T.copy())
        return np.array(self.time_hist)


    def plot_snapshot(self, idx: int = -1):
        plt.figure(figsize=(5, 4))
        temp_C = self.time_hist[idx] - 273.15
        plt.imshow(temp_C, origin="lower", extent=[0, self.W*1000, 0, self.H*1000], cmap="hot")
        plt.colorbar(label="T / °C")
        plt.xlabel("mm"); plt.ylabel("mm")
        t = idx * self.dt
        plt.title(f"2‑D Temperature Map (t={t:.0f}s)")
        plt.tight_layout(); Path("results").mkdir(exist_ok=True)
        plt.savefig(f"results/thermal_2d_snapshot_{idx}.png", dpi=300)
        plt.show()





def main():
    Path("results").mkdir(exist_ok=True)
    start = time.time()
    # ---------------- 1‑D ----------------
    solver1d = Solver1D(
        thickness=1e-3,
        nx=60,
        dt=1e-4,
        t_total=20,
        q_dot=5000,
        bc_left=(BCType.FIXED, 298.15),
        bc_right=(BCType.CONVECT, (10, 298.15)),
    )


    hist_1d = solver1d.run(record_interval=5)
    solver1d.plot_profiles()


    solver2d = Solver2D(
        width=0.02,
        height=0.02,
        nx=80,
        ny=80,
        dt=0.02,
        t_total=100,
        q_dot=5000,
        bc_type=BCType.FIXED,
        bc_value=298.15,
    )
    solver2d.run(record_interval=2)
    solver2d.plot_snapshot(idx=-1)


    np.savez(
        "results/thermal_simulation.npz",
        x_1d=solver1d.x,
        t_hist_1d=np.arange(len(solver1d.time_hist))*5,
        temp_hist_1d=hist_1d,
        temp_final_2d=solver2d.time_hist[-1],
        x_2d=solver2d.x,
        y_2d=solver2d.y,
    )
    print("仿真完成，总耗时 %.2f s" % (time.time()-start))

if __name__ == "__main__":
    main()
