
# -*- coding: utf-8 -*-
"""
PID 温度控制 + PSO 参数优化（中文注释版）
------------------------------------------------
脚本功能：
1. 从阶跃实验数据（CSV）进行一阶惯性系统辨识。
2. 使用 Ziegler‑Nichols 方法给出 PID 初始参数。
3. 用粒子群算法 (PSO) 全局搜索最优 PID 参数，使超调量、稳态误差、调节时间的加权和最小。
4. 闭环仿真对比初始 PID 与优化 PID，并绘制温度响应曲线。

依赖库：pandas  numpy  matplotlib  scipy
安装命令：pip install pandas numpy matplotlib scipy
作者：ChatGPT-o3  (示例代码，如需修改请自便)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Tuple

def load_step_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取阶跃实验 CSV，返回时间、温度、加热电压三个 ndarray"""
    df = pd.read_csv(csv_path)
    t = df.iloc[:, 0].to_numpy(float)
    temp = df.iloc[:, 1].to_numpy(float)
    volt = df.iloc[:, 2].to_numpy(float)
    return t, temp, volt

def fopdt_step(t: np.ndarray, K: float, tau: float, u_step: float, y0: float) -> np.ndarray:
    """一阶惯性（无纯滞后）对阶跃输入的响应"""
    return y0 + K * u_step * (1 - np.exp(-t / tau))

def identification_cost(params: np.ndarray, t, y, u_step, y0):
    """辨识目标函数：均方误差"""
    K, tau = params
    y_hat = fopdt_step(t, K, tau, u_step, y0)
    return np.mean((y - y_hat) ** 2)

def simulate_pid(t, setpoint, K, tau, Kp, Ki, Kd, u_min=0.0, u_max=3.5):
    """基于一阶模型的离散 PID 闭环仿真"""
    dt = np.diff(t, prepend=t[0])
    y = np.zeros_like(t)
    integral = 0.0
    e_prev = 0.0
    y[0] = setpoint[0]  # 初始温度
    for i in range(1, len(t)):
        error = setpoint[i] - y[i-1]
        integral += error * dt[i]
        derivative = (error - e_prev) / dt[i] if dt[i] > 0 else 0.0
        # PID 输出
        u = Kp * error + Ki * integral + Kd * derivative
        # 饱和 + 抗积分饱和
        u = np.clip(u, u_min, u_max)
        if (u == u_min or u == u_max) and np.sign(error) == np.sign(integral):
            integral -= error * dt[i]
        # 一阶模型更新
        y[i] = y[i-1] + (dt[i] / tau) * (-y[i-1] + (K * u))
        e_prev = error
    return y

def performance_metrics(t, y, setpoint):
    """返回超调量、稳态误差、调节时间"""
    sp = setpoint[-1]
    Mp = max(0, y.max() - sp)              # 超调
    ess = abs(y[-1] - sp)                  # 稳态误差
    tol = 0.02 * sp                        # ±2% 带
    ts = t[-1]
    for ti, yi in zip(t, y):
        if abs(yi - sp) <= tol and np.all(np.abs(y[t >= ti] - sp) <= tol):
            ts = ti
            break
    return Mp, ess, ts

def zn_pid(K, tau):
    """Z-N 阶跃响应法近似整定（这里按 Lambda=τ 取系数）"""
    Kp = 0.6 * tau / (K * tau)  # 简化示例
    Ti = tau
    Td = 0.125 * tau
    Ki = Kp / Ti
    Kd = Kp * Td
    return Kp, Ki, Kd

def pso_optimize(objective, bounds, n_particles=30, iters=50, w=0.7, c1=1.4, c2=1.4):
    """简单粒子群优化实现"""
    dim = len(bounds)
    # 初始化粒子位置与速度
    pos = np.array([np.random.uniform(b[0], b[1], dim) for _ in range(n_particles)])
    vel = np.zeros_like(pos)
    p_best = pos.copy()
    p_best_val = np.array([objective(p) for p in pos])
    g_best = p_best[p_best_val.argmin()].copy()

    for _ in range(iters):
        r1, r2 = np.random.rand(n_particles, dim), np.random.rand(n_particles, dim)
        vel = w * vel + c1 * r1 * (p_best - pos) + c2 * r2 * (g_best - pos)
        pos += vel
        # 越界处理
        for i in range(dim):
            pos[:, i] = np.clip(pos[:, i], bounds[i][0], bounds[i][1])
        # 评价
        vals = np.array([objective(p) for p in pos])
        improve = vals < p_best_val
        p_best[improve] = pos[improve]
        p_best_val[improve] = vals[improve]
        # 更新全局最优
        if vals.min() < performance := objective(g_best):
            g_best = pos[vals.argmin()].copy()
    return g_best

def main():
    # === 读取数据 ===
    csv_path = 'B 任务数据集.csv'
    t, temp, volt = load_step_data(csv_path)
    u_step = volt.max() - volt[0]
    y0 = temp[0]

    # === 一阶模型辨识 ===
    res = minimize(identification_cost,
                   x0=[10.0, 2000.0],
                   args=(t, temp, u_step, y0),
                   bounds=[(0.1, 100), (100, 1e5)],
                   method='L-BFGS-B')
    K_hat, tau_hat = res.x
    print(f'辨识结果: K={K_hat:.2f}, tau={tau_hat:.1f}s')

    # === Z-N 初始 PID ===
    Kp0, Ki0, Kd0 = zn_pid(K_hat, tau_hat)
    print('Z-N 初始 PID:', Kp0, Ki0, Kd0)

    # === PSO 优化 PID ===
    setpoint = np.ones_like(t) * 40.0
    def obj(params):
        Kp, Ki, Kd = params
        y_sim = simulate_pid(t, setpoint, K_hat, tau_hat, Kp, Ki, Kd)
        Mp, ess, ts = performance_metrics(t, y_sim, setpoint)
        return Mp/setpoint[-1] + ess/setpoint[-1] + ts/t[-1]

    bounds = [(0, 10), (0, 0.2), (0, 300)]
    Kp_opt, Ki_opt, Kd_opt = pso_optimize(obj, bounds)
    print('PSO 优化 PID:', Kp_opt, Ki_opt, Kd_opt)

    # === 仿真对比 ===
    y_init = simulate_pid(t, setpoint, K_hat, tau_hat, Kp0, Ki0, Kd0)
    y_opt = simulate_pid(t, setpoint, K_hat, tau_hat, Kp_opt, Ki_opt, Kd_opt)

    print('初始 PID 指标:', performance_metrics(t, y_init, setpoint))
    print('优化 PID 指标:', performance_metrics(t, y_opt, setpoint))

    # === 绘图 ===
    plt.figure(figsize=(8, 4))
    plt.plot(t/60, y_init, label='初始 PID')
    plt.plot(t/60, y_opt, label='PSO 优化 PID')
    plt.axhline(40, color='k', linestyle='--', label='设定值 40℃')
    plt.xlabel('时间 (分钟)')
    plt.ylabel('温度 (℃)')
    plt.title('闭环温度响应对比')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
