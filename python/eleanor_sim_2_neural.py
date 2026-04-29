"""
ELEANOR v4.0 模拟实验 — 第二部分
模块3: 神经刺激仿真 (40Hz / 140Hz)
直接运行: python eleanor_sim_2_neural.py
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
#  模块3: 神经刺激仿真
# ═══════════════════════════════════════════════════════════════

# ---------- 3A: 40Hz Gamma脑波夹带 ----------

def simulate_gamma_entrainment(n_neurons=200, stim_freq=40.0, sim_time=5.0, dt=0.001):
    np.random.seed(42)
    N = n_neurons
    steps = int(sim_time / dt)
    t = np.arange(steps) * dt
    omega_healthy = np.random.normal(40, 3, N)
    omega_ad      = np.random.normal(40, 12, N)
    K = 5.0
    A_stim = 15.0

    def run(omega):
        theta = np.random.uniform(0, 2*np.pi, N)
        r = np.zeros(steps)
        for i in range(steps):
            r[i] = np.abs(np.mean(np.exp(1j * theta)))
            stim = A_stim * np.sin(2 * np.pi * stim_freq * t[i]) if t[i] > 0.5 else 0
            coupling = (K / N) * np.sum(np.sin(theta - theta[:, None]), axis=1)
            theta = theta + (omega + coupling + stim) * dt
        return r

    return t, run(omega_healthy), run(omega_ad)


def run_module_3A():
    print("=" * 65)
    print("  模块3A: 40Hz Gamma脑波夹带仿真")
    print("=" * 65)

    t, r_healthy, r_ad = simulate_gamma_entrainment()

    r_h_pre  = np.mean(r_healthy[t < 0.5])
    r_h_post = np.mean(r_healthy[t > 3.0])
    r_a_pre  = np.mean(r_ad[t < 0.5])
    r_a_post = np.mean(r_ad[t > 3.0])

    print(f"  健康基线: 刺激前同步性={r_h_pre:.3f} → 刺激后={r_h_post:.3f}")
    print(f"  AD患者:   刺激前同步性={r_a_pre:.3f} → 刺激后={r_a_post:.3f}")
    print(f"  AD患者同步性提升: {(r_a_post - r_a_pre)/r_a_pre*100:.1f}%")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    ax1.plot(t, r_healthy, 'b-', alpha=0.6, label='健康基线')
    ax1.plot(t, r_ad, 'r-', alpha=0.6, label='AD患者')
    ax1.axvline(0.5, color='green', ls='--', lw=2, label='刺激开始')
    ax1.set(xlabel='时间', ylabel='Kuramoto序参量 r',
            title='40Hz Gamma夹带: Kuramoto同步性', ylim=(0, 1.05))
    ax1.legend()

    win = 200
    kernel = np.ones(win) / win
    ax2.plot(t, np.convolve(r_healthy, kernel, mode='same'), 'b-', lw=2, label='健康基线(平滑)')
    ax2.plot(t, np.convolve(r_ad, kernel, mode='same'), 'r-', lw=2, label='AD患者(平滑)')
    ax2.axvline(0.5, color='green', ls='--', lw=2, label='刺激开始')
    ax2.set(xlabel='时间', ylabel='同步性 r (滑动平均)',
            title='40Hz Gamma夹带: 平滑曲线', ylim=(0, 1.05))
    ax2.legend()
    plt.tight_layout()
    plt.savefig('sim_3A_gamma_entrainment.png', dpi=150)
    print("  📊 图已保存: sim_3A_gamma_entrainment.png\n")
    plt.close()


# ---------- 3B: 140Hz 高频刺激效应 ----------

def simulate_high_freq(stim_freq=140.0, sim_time=2.0, dt=0.0005):
    steps = int(sim_time / dt)
    t = np.arange(steps) * dt
    np.random.seed(42)

    tau_m = 0.02
    V_rest = -65.0
    V_thresh = -50.0
    V_reset = -70.0
    R_m = 10.0

    def run_with_stim(freq, amp):
        V = np.full(steps, V_rest)
        spikes = np.zeros(steps)
        I_stim = amp * np.sin(2 * np.pi * freq * t)
        I_stim[t < 0.5] = 0
        I_noise = np.random.normal(0, 1.5, steps)
        for i in range(1, steps):
            dV = (-(V[i-1] - V_rest) + R_m * (I_stim[i] + I_noise[i])) / tau_m * dt
            V[i] = V[i-1] + dV
            if V[i] >= V_thresh:
                V[i] = V_reset
                spikes[i] = 1
        return V, spikes

    return (t, *run_with_stim(0, 0), *run_with_stim(140, 8.0), *run_with_stim(40, 8.0))


def run_module_3B():
    print("=" * 65)
    print("  模块3B: 140Hz 高频刺激效应仿真")
    print("=" * 65)

    t, V_base, sp_base, V_140, sp_140, V_40, sp_40 = simulate_high_freq()

    def firing_rate(spikes, t_arr, t_start, t_end):
        mask = (t_arr >= t_start) & (t_arr < t_end)
        return np.sum(spikes[mask]) / (t_end - t_start)

    fr_base = firing_rate(sp_base, t, 0.5, 2.0)
    fr_40   = firing_rate(sp_40, t, 0.5, 2.0)
    fr_140  = firing_rate(sp_140, t, 0.5, 2.0)

    print(f"  无刺激放电率: {fr_base:.1f} Hz")
    print(f"  40Hz刺激放电率: {fr_40:.1f} Hz (变化: {(fr_40-fr_base)/max(fr_base,0.1)*100:+.1f}%)")
    print(f"  140Hz刺激放电率: {fr_140:.1f} Hz (变化: {(fr_140-fr_base)/max(fr_base,0.1)*100:+.1f}%)")

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, V, sp, title, color in [
        (axes[0], V_base, sp_base, '无刺激基线', 'gray'),
        (axes[1], V_40, sp_40, '40Hz刺激', 'blue'),
        (axes[2], V_140, sp_140, '140Hz刺激', 'red'),
    ]:
        ax.plot(t, V, color=color, alpha=0.7, lw=0.5)
        sp_times = t[sp > 0.5]
        ax.scatter(sp_times, np.full_like(sp_times, -45), color=color, s=5, zorder=5)
        ax.axvline(0.5, color='green', ls='--', lw=1.5)
        ax.set(ylabel='膜电位', title=title, ylim=(-80, -40))
    axes[2].set(xlabel='时间
    plt.tight_layout()
    plt.savefig('sim_3B_high_freq_stimulation.png', dpi=150)
    print("  📊 图已保存: sim_3B_high_freq_stimulation.png\n")
    plt.close()


# ---------- 3C: 感觉神经元-脂质体频率特异性融合 ----------

def run_module_3C():
    print("=" * 65)
    print("  模块3C: 感觉神经元-脂质体频率特异性融合")
    print("=" * 65)

    freqs = np.linspace(1, 250, 500)

    def resonance_curve(f, f_res, zeta, f_cutoff):
        r = f / f_res
        H_mag = r / np.sqrt((1 - r**2)**2 + (2 * zeta * r)**2)
        W = 1.0 / (1.0 + (f / f_cutoff)**2)
        return H_mag**2 * W

    eta_A = resonance_curve(freqs, f_res=40, zeta=0.15, f_cutoff=200)
    eta_B = resonance_curve(freqs, f_res=140, zeta=0.20, f_cutoff=500)

    eta_A_norm = eta_A / np.max(eta_A) * 100
    eta_B_norm = eta_B / np.max(eta_B) * 100

    def eff_at(freq, curve):
        return curve[np.argmin(np.abs(freqs - freq))]

    print(f"\n  {'频率':<10} {'类型A融合效率':<18} {'类型B融合效率':<18} {'说明'}")
    print("  " + "-" * 70)
    for f_check in [10, 20, 40, 60, 80, 100, 120, 140, 160, 200]:
        eA = eff_at(f_check, eta_A_norm)
        eB = eff_at(f_check, eta_B_norm)
        note = ""
        if f_check == 40:
            note = "← 类型A共振峰 (ELEANOR设计频率)"
        elif f_check == 140:
            note = "← 类型B共振峰 (ELEANOR设计频率)"
        elif f_check in [60, 80, 100, 120]:
            note = "非共振, 融合效率低"
        print(f"  {f_check:<10} {eA:<18.1f} {eB:<18.1f} {note}")

    sel_40 = eff_at(40, eta_A_norm) / max(eff_at(40, eta_B_norm), 0.01)
    sel_140 = eff_at(140, eta_B_norm) / max(eff_at(140, eta_A_norm), 0.01)
    print(f"\n  40Hz对类型A的选择性倍数: {sel_40:.1f}x")
    print(f"  140Hz对类型B的选择性倍数: {sel_140:.1f}x")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(freqs, eta_A_norm, 'b-', lw=2.5, label='类型A感受器 (共振≈40Hz)')
    ax1.plot(freqs, eta_B_norm, 'r-', lw=2.5, label='类型B感受器 (共振≈140Hz)')
    ax1.axvline(40, color='blue', ls='--', lw=1.5, alpha=0.7)
    ax1.axvline(140, color='red', ls='--', lw=1.5, alpha=0.7)
    ax1.scatter([40], [eff_at(40, eta_A_norm)], color='blue', s=120, zorder=5, marker='*')
    ax1.scatter([140], [eff_at(140, eta_B_norm)], color='red', s=120, zorder=5, marker='*')
    ax1.text(42, 103, '40Hz\n(设计频率)', fontsize=9, color='blue', fontweight='bold')
    ax1.text(142, 103, '140Hz\n(设计频率)', fontsize=9, color='red', fontweight='bold')
    ax1.set(xlabel='刺激频率', ylabel='融合效率 (归一化%)',
            title='感觉神经元-脂质体融合频率调谐曲线', xlim=(0, 250), ylim=(0, 115))
    ax1.legend(fontsize=10)

    t_demo = np.linspace(0, 0.1, 1000)

    def membrane_oscillation(t, f_stim, f_res, zeta):
        r = f_stim / f_res
        H_mag = r / np.sqrt((1 - r**2)**2 + (2 * zeta * r)**2)
        phase = np.arctan2(-2 * zeta * r, 1 - r**2)
        return H_mag * np.sin(2 * np.pi * f_stim * t + phase)

    v_40  = membrane_oscillation(t_demo, 40, 40, 0.15)
    v_80  = membrane_oscillation(t_demo, 80, 40, 0.15)
    v_140 = membrane_oscillation(t_demo, 140, 140, 0.20)
    v_60  = membrane_oscillation(t_demo, 60, 140, 0.20)

    v_40  = v_40 / max(abs(v_40.max()), abs(v_40.min())) * 15
    v_80  = v_80 / max(abs(v_80.max()), abs(v_80.min())) * 15
    v_140 = v_140 / max(abs(v_140.max()), abs(v_140.min())) * 15
    v_60  = v_60 / max(abs(v_60.max()), abs(v_60.min())) * 15

    ax2.plot(t_demo * 1000, v_40, 'b-', lw=2, label='类型A @ 40Hz (共振)')
    ax2.plot(t_demo * 1000, v_80, 'b--', lw=1.5, alpha=0.5, label='类型A @ 80Hz (失谐)')
    ax2.plot(t_demo * 1000, v_140, 'r-', lw=2, label='类型B @ 140Hz (共振)')
    ax2.plot(t_demo * 1000, v_60, 'r--', lw=1.5, alpha=0.5, label='类型B @ 60Hz (失谐)')
    ax2.axhline(0, color='gray', lw=0.5)
    ax2.set(xlabel='时间', ylabel='膜电位偏移 (mV, 归一化)',
            title='亚阈值膜电位振荡: 共振 vs 失谐', xlim=(0, 100))
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('sim_3C_fusion_frequency_specificity.png', dpi=150)
    print("\n  📊 图已保存: sim_3C_fusion_frequency_specificity.png\n")
    plt.close()


# ---------- 3D: 带宽上限硬约束验证 ----------

def run_module_3D():
    print("=" * 65)
    print("  模块3D: 带宽上限硬约束验证 (不超过病前基线)")
    print("=" * 65)

    fs = 1000
    duration = 5.0
    t = np.arange(0, duration, 1/fs)
    BW_LOWER = 30
    BW_UPPER = 80

    stim_40_bounded = np.sin(2 * np.pi * 40 * t)
    np.random.seed(42)
    stim_unbounded = np.sin(2 * np.pi * 40 * t) + 0.5 * np.sin(2 * np.pi * 120 * t) + 0.3 * np.sin(2 * np.pi * 200 * t)

    def compute_bandwidth_power(signal, f_low, f_high):
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1/fs)
        power = np.abs(fft) ** 2
        total_power = np.sum(power)
        mask = (freqs >= f_low) & (freqs <= f_high)
        band_power = np.sum(power[mask])
        out_power = np.sum(power[~mask])
        return band_power / total_power * 100, out_power / total_power * 100

    bp_40, op_40 = compute_bandwidth_power(stim_40_bounded, BW_LOWER, BW_UPPER)
    bp_unb, op_unb = compute_bandwidth_power(stim_unbounded, BW_LOWER, BW_UPPER)

    print(f"  安全带宽窗口: {BW_LOWER}-{BW_UPPER} Hz")
    print(f"  ELEANOR 40Hz: 窗口内={bp_40:.1f}%, 窗口外={op_40:.1f}%  → {'✅ 合规' if bp_40 > 95 else '❌ 违规'}")
    print(f"  不受限刺激:   窗口内={bp_unb:.1f}%, 窗口外={op_unb:.1f}%  → {'✅ 合规' if bp_unb > 95 else '❌ 违规'}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    for ax, sig, title in [(ax1, stim_40_bounded, 'ELEANOR 40Hz (带限)'),
                            (ax2, stim_unbounded, '不受限宽频刺激 (违规)')]:
        fft = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(len(sig), 1/fs)
        power = 20 * np.log10(np.abs(fft) + 1e-10)
        ax.plot(freqs, power, 'b-', lw=0.8)
        ax.axvspan(BW_LOWER, BW_UPPER, alpha=0.15, color='green', label=f'安全窗口 {BW_LOWER}-{BW_UPPER}Hz')
        ax.axvline(BW_LOWER, color='green', ls='--', lw=1)
        ax.axvline(BW_UPPER, color='green', ls='--', lw=1)
        ax.set(xlabel='频率', ylabel='功率', title=title,
               xlim=(0, 250), ylim=(-40, 60))
        ax.legend()

    plt.tight_layout()
    plt.savefig('sim_3D_bandwidth_constraint.png', dpi=150)
    print("  📊 图已保存: sim_3D_bandwidth_constraint.png\n")
    plt.close()


# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "▓" * 65)
    print("  ELEANOR v4.0 模拟实验 — Part 2: 神经刺激仿真")
    print("▓" * 65 + "\n")

    run_module_3A()
    run_module_3B()
    run_module_3C()
    run_module_3D()

    print("▓" * 65)
    print("  Part 2 全部完成")
    print("▓" * 65)
