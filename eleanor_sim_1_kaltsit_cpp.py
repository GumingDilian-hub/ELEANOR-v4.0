"""
ELEANOR v4.0 模拟实验 — 第一部分
模块1: KALTSIT 分子AND门动力学
模块2: CPP递送 & 酶降解 PK/PD
直接运行: python eleanor_sim_1_kaltsit_cpp.py
"""

import numpy as np
from scipy.integrate import solve_ivp
from itertools import product
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
#  模块1: KALTSIT 分子AND门
# ═══════════════════════════════════════════════════════════════

class KALTSIT_AND_Gate:
    """
    DNA strand displacement ODE 模型
    物种: W(Wire游离), A(Anchor游离), WE(完美组装), Ws(错误中间体), Deg(降解)
    """
    def __init__(self):
        self.k_bind     = 1e6    # M⁻¹s⁻¹ 正确组装
        self.k_err      = 5e4    # M⁻¹s⁻¹ 错误组装
        self.k_dissolve = 0.05   # s⁻¹     自溶
        self.k_stable   = 1e-6   # s⁻¹     存活态降解(≈0)
        self.W0 = 100e-9
        self.A0 = 100e-9

    def ode(self, t, y, and_ok):
        W, Anc, WE, Ws, Deg = y
        if and_ok:
            r = self.k_bind * W * Anc
            return [-r, -r, r - self.k_stable*WE, 0, self.k_stable*WE]
        else:
            rb = self.k_err * W * Anc
            rd = self.k_dissolve * Ws
            return [-rb, -rb, 0, rb - rd, rd]

    def run(self, and_ok=True, t_max=600, n=1000):
        return solve_ivp(self.ode, (0, t_max),
                         [self.W0, self.A0, 0, 0, 0],
                         args=(and_ok,),
                         t_eval=np.linspace(0, t_max, n),
                         method='LSODA', rtol=1e-10, atol=1e-15)


def run_module_1A():
    print("=" * 65)
    print("  模块1A: KALTSIT AND门 ODE动力学")
    print("=" * 65)
    m = KALTSIT_AND_Gate()
    sol_ok   = m.run(and_ok=True,  t_max=600)
    sol_fail = m.run(and_ok=False, t_max=600)

    t10 = np.argmin(np.abs(sol_ok.t - 600))
    survive = sol_ok.y[2, t10] / m.W0 * 100
    deg    = sol_fail.y[4, -1] / m.W0 * 100
    print(f"  ✅ AND满足:   10min存活率 = {survive:.1f}%")
    print(f"  ❌ AND不满足: 10min降解率 = {deg:.1f}%")

    # 自溶半衰期
    pk = np.argmax(sol_fail.y[3])
    Ws_peak = sol_fail.y[3, pk]
    post = sol_fail.y[3, pk:]
    half = Ws_peak / 2
    if np.any(post < half):
        idx = np.argmax(post < half)
        t_half = sol_fail.t[pk + idx] - sol_fail.t[pk]
        print(f"  ❌ 错误组装体自溶半衰期 = {t_half:.1f} 秒")

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    labels = ['Wire', 'Anchor', '组装体(存活)', '错误中间体', '降解产物']
    for i, lb in enumerate(labels):
        ax1.plot(sol_ok.t / 60, sol_ok.y[i] / 1e-9, label=lb)
    ax1.set(xlabel='时间', ylabel='浓度', title='AND条件满足 → 存活', ylim=(0, 110))
    ax1.legend(fontsize=8)

    for i, lb in enumerate(labels):
        ax2.plot(sol_fail.t / 60, sol_fail.y[i] / 1e-9, label=lb)
    ax2.set(xlabel='时间', ylabel='浓度', title='AND条件不满足 → 自溶', ylim=(0, 110))
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('sim_1A_kaltsit_ode.png', dpi=150)
    print("  📊 图已保存: sim_1A_kaltsit_ode.png\n")
    plt.close()


def run_module_1B():
    """
    KALTSIT AND门在生物噪声下的鲁棒性分析
    ─────────────────────────────────────
    细胞内分子数有限(100nM在fL体积内仅~60个分子),
    存在显著的泊松浓度涨落。用Euler-Maruyama方法
    求解随机ODE(SDE), 统计不同噪声强度下AND门的误判率。
    """
    print("=" * 65)
    print("  模块1B: KALTSIT AND门 生物噪声鲁棒性分析")
    print("=" * 65)

    m = KALTSIT_AND_Gate()
    t_max = 600
    dt = 0.5
    n_steps = int(t_max / dt)
    n_mc = 500

    noise_levels = np.array([0.0, 0.05, 0.10, 0.13, 0.20, 0.30, 0.50])
    labels_noise = ['0%', '5%', '10%', '13%(生理)', '20%', '30%', '50%']

    thresh_survive = 0.5 * m.W0
    thresh_degrade = 0.5 * m.W0

    rate_ok_list = []
    rate_fail_list = []

    for ni, noise in enumerate(noise_levels):
        survive_ok = 0
        degrade_ok = 0

        for mc in range(n_mc):
            np.random.seed(mc + ni * 10000)

            # ── 场景1: AND满足 ──
            W, A, WE = m.W0, m.A0, 0.0
            for _ in range(n_steps):
                r = m.k_bind * W * A
                W  += -r * dt
                A  += -r * dt
                WE += (r - m.k_stable * WE) * dt
                sq_dt = np.sqrt(dt)
                W  += noise * np.sqrt(max(W, 0)) * np.random.normal() * sq_dt
                A  += noise * np.sqrt(max(A, 0)) * np.random.normal() * sq_dt
                WE += noise * np.sqrt(max(WE, 0)) * np.random.normal() * sq_dt * 0.1
                W  = max(W, 0)
                A  = max(A, 0)
                WE = max(WE, 0)

            if WE > thresh_survive:
                survive_ok += 1

            # ── 场景2: AND不满足 ──
            W, A, Ws, Deg = m.W0, m.A0, 0.0, 0.0
            for _ in range(n_steps):
                rb = m.k_err * W * A
                rd = m.k_dissolve * Ws
                W   += -rb * dt
                A   += -rb * dt
                Ws  += (rb - rd) * dt
                Deg += rd * dt
                sq_dt = np.sqrt(dt)
                W   += noise * np.sqrt(max(W, 0)) * np.random.normal() * sq_dt
                A   += noise * np.sqrt(max(A, 0)) * np.random.normal() * sq_dt
                Ws  += noise * np.sqrt(max(Ws, 0)) * np.random.normal() * sq_dt
                W   = max(W, 0)
                A   = max(A, 0)
                Ws  = max(Ws, 0)
                Deg = max(Deg, 0)

            if Deg > thresh_degrade:
                degrade_ok += 1

        r_ok = survive_ok / n_mc * 100
        r_fail = degrade_ok / n_mc * 100
        rate_ok_list.append(r_ok)
        rate_fail_list.append(r_fail)

        tag = " ← 生理噪声" if "生理" in labels_noise[ni] else ""
        print(f"  σ={labels_noise[ni]:<12} AND满足→正确存活={r_ok:6.1f}%  "
              f"AND不满足→正确自溶={r_fail:6.1f}%{tag}")

    phys_idx = 3
    false_neg = 100 - rate_ok_list[phys_idx]
    false_pos = 100 - rate_fail_list[phys_idx]
    print(f"\n  生理噪声(13%)下:")
    print(f"    假阴性率(该存活判为自溶): {false_neg:.1f}%")
    print(f"    假阳性率(该自溶判为存活): {false_pos:.1f}%")
    print(f"    总误判率: {(false_neg + false_pos)/2:.1f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(noise_levels * 100, rate_ok_list, 'bo-', lw=2, ms=8, label='AND满足→正确存活')
    ax1.plot(noise_levels * 100, rate_fail_list, 'rs-', lw=2, ms=8, label='AND不满足→正确自溶')
    ax1.axvspan(5, 20, alpha=0.1, color='green', label='生理噪声范围(5-20%)')
    ax1.axvline(13, color='green', ls=':', lw=1.5)
    ax1.set(xlabel='浓度噪声强度 σ/μ (%)', ylabel='正确判定率 (%)',
            title='AND门噪声鲁棒性', ylim=(0, 105))
    ax1.legend(fontsize=9)

    np.random.seed(42)
    noise_demo = 0.13
    W_traj_ok, W_traj_fail = [], []
    W, A, WE = m.W0, m.A0, 0.0
    for _ in range(n_steps):
        r = m.k_bind * W * A
        W += -r * dt + noise_demo * np.sqrt(max(W, 0)) * np.random.normal() * np.sqrt(dt)
        W = max(W, 0)
        W_traj_ok.append(W)

    W, A, Ws, Deg = m.W0, m.A0, 0.0, 0.0
    for _ in range(n_steps):
        rb = m.k_err * W * A
        rd = m.k_dissolve * Ws
        W += -rb * dt + noise_demo * np.sqrt(max(W, 0)) * np.random.normal() * np.sqrt(dt)
        Ws += (rb - rd) * dt + noise_demo * np.sqrt(max(Ws, 0)) * np.random.normal() * np.sqrt(dt)
        W = max(W, 0)
        Ws = max(Ws, 0)
        W_traj_fail.append(W)

    t_traj = np.arange(n_steps) * dt / 60
    ax2.plot(t_traj, np.array(W_traj_ok)/1e-9, 'b-', alpha=0.6, label='AND满足: Wire浓度')
    ax2.plot(t_traj, np.array(W_traj_fail)/1e-9, 'r-', alpha=0.6, label='AND不满足: Wire浓度')
    ax2.set(xlabel='时间', ylabel='Wire浓度',
            title=f'噪声轨迹示例 (σ={noise_demo:.0%}, 1条代表性轨迹)')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('sim_1B_noise_robustness.png', dpi=150)
    print("  📊 图已保存: sim_1B_noise_robustness.png\n")
    plt.close()


def run_module_1C():
    print("=" * 65)
    print("  模块1C: 错误组装→自溶 半衰期 & 残留分析")
    print("=" * 65)
    scenarios = {
        "单链暴露(快速)":   0.10,
        "部分组装(中速)":   0.05,
        "拓扑错配(慢速)":   0.02,
    }
    print(f"  {'场景':<18} {'k_deg(s⁻¹)':<12} {'理论t½(s)':<12} {'5min残留%':<12} {'10min残留%':<12}")
    print("  " + "-" * 66)

    t_eval = np.linspace(0, 600, 2000)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, k in scenarios.items():
        sol = solve_ivp(lambda t, y: [-k * y[0]], (0, 600), [100.0],
                        t_eval=t_eval, method='LSODA')
        t_half = np.log(2) / k
        r5  = sol.y[0, np.argmin(np.abs(sol.t - 300))]
        r10 = sol.y[0, -1]
        print(f"  {name:<18} {k:<12.3f} {t_half:<12.1f} {r5:<12.2f} {r10:<12.2f}")
        ax.plot(sol.t / 60, sol.y[0], label=f"{name} (t½={t_half:.0f}s)")

    ax.axhline(1, color='red', ls='--', lw=1, label='1% 残留阈值')
    ax.set(xlabel='时间', ylabel='错误组装体残留 (%)',
           title='不同错误类型的自溶曲线', ylim=(0, 105))
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('sim_1C_autolysis_curves.png', dpi=150)
    print("  📊 图已保存: sim_1C_autolysis_curves.png\n")
    plt.close()


# ═══════════════════════════════════════════════════════════════
#  模块2: CPP递送 & 酶降解 PK/PD
# ═══════════════════════════════════════════════════════════════

class CPP_Delivery_Model:
    """
    三阶段PK模型:
      Phase 1 (0-10min):   CPP-DNA复合物跨膜内吞
      Phase 2 (10-15min):  DNase I/II 降解未修饰C段
      Phase 3 (15-40min):  26S蛋白酶体降解游离CPP, t½≈26s
    """
    def __init__(self):
        self.k_endocytosis = 0.005
        self.CPP_DNA_out0  = 100.0
        self.k_dnase = 0.01
        self.k_proteasome = np.log(2) / 26.0

    def ode_full(self, t, y):
        ext, cyt, c_free, cpp_free, deg_c, deg_cpp = y
        r_endo = self.k_endocytosis * ext
        r_release_c = 0.3 * r_endo
        r_dnase = self.k_dnase * c_free
        r_protea = self.k_proteasome * cpp_free
        return [
            -r_endo,
            r_endo - r_release_c,
            r_release_c - r_dnase,
            r_release_c - r_protea,
            r_dnase,
            r_protea
        ]

    def run(self, t_max=2400, n=3000):
        return solve_ivp(self.ode_full, (0, t_max),
                         [self.CPP_DNA_out0, 0, 0, 0, 0, 0],
                         t_eval=np.linspace(0, t_max, n),
                         method='LSODA', rtol=1e-10, atol=1e-12)


def run_module_2():
    print("=" * 65)
    print("  模块2: CPP递送 & 酶降解 PK/PD 仿真")
    print("=" * 65)
    m = CPP_Delivery_Model()
    print(f"  参数: k_endo={m.k_endocytosis} s⁻¹, k_dnase={m.k_dnase} s⁻¹, "
          f"k_proteasome={m.k_proteasome:.4f} s⁻¹ (t½=26s)")
    print()

    sol = m.run(t_max=2400)
    t = sol.t / 60

    checkpoints = {'0min': 0, '5min': 5, '10min': 10, '15min': 15, '20min': 20, '30min': 30, '40min': 40}
    print(f"  {'时间点':<10} {'胞外(%)':<12} {'胞质内(%)':<12} {'C段游离(%)':<12} {'CPP游离(%)':<12} {'C降解(%)':<12} {'CPP降解(%)':<12}")
    print("  " + "-" * 82)

    for label, tmin in checkpoints.items():
        idx = np.argmin(np.abs(t - tmin))
        vals = sol.y[:, idx]
        total = m.CPP_DNA_out0
        row = [f"{v/total*100:>8.2f}" for v in vals]
        print(f"  {label:<10} " + "  ".join(row))

    cpp_peak = np.max(sol.y[3])
    if cpp_peak > 0.1:
        pk_idx = np.argmax(sol.y[3])
        half = cpp_peak / 2
        post = sol.y[3, pk_idx:]
        if np.any(post < half):
            h_idx = np.argmax(post < half)
            t_half_actual = sol.t[pk_idx + h_idx] - sol.t[pk_idx]
            print(f"\n  📐 CPP实际降解半衰期 = {t_half_actual:.1f} 秒 (设计值: 26秒)")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    labels_pk = ['胞外CPP-DNA', '胞质CPP-DNA', 'C段游离', 'CPP游离', 'C降解产物', 'CPP降解产物']
    for i, lb in enumerate(labels_pk):
        ax.plot(t, sol.y[i], label=lb)
    ax.set(xlabel='时间', ylabel='浓度', title='CPP递送全程PK曲线')
    ax.legend(fontsize=7)
    ax.axvspan(0, 10, alpha=0.08, color='blue')
    ax.axvspan(10, 15, alpha=0.08, color='orange')
    ax.axvspan(15, 40, alpha=0.08, color='red')
    ax.text(5, 90, 'Phase 1\n跨膜内吞', ha='center', fontsize=9, color='blue')
    ax.text(12.5, 90, 'Phase 2\nDNase降解', ha='center', fontsize=9, color='orange')
    ax.text(27.5, 90, 'Phase 3\n蛋白酶体降解', ha='center', fontsize=9, color='red')

    ax = axes[0, 1]
    ext_pct = sol.y[0] / m.CPP_DNA_out0 * 100
    cyt_pct = sol.y[1] / m.CPP_DNA_out0 * 100
    ax.plot(t, ext_pct, 'b-', label='胞外残留')
    ax.plot(t, cyt_pct, 'g-', label='胞质内')
    ax.fill_between(t, 0, cyt_pct, alpha=0.15, color='green')
    ax.set(xlabel='时间', ylabel='百分比(%)', title='Phase 1: 跨膜内吞 (0-10min)',
           xlim=(0, 15), ylim=(0, 105))
    ax.legend()

    ax = axes[1, 0]
    c_pct = sol.y[2] / m.CPP_DNA_out0 * 100
    dc_pct = sol.y[4] / m.CPP_DNA_out0 * 100
    ax.plot(t, c_pct, 'r-', label='C段游离')
    ax.plot(t, dc_pct, 'm--', label='C段已降解')
    ax.set(xlabel='时间', ylabel='百分比(%)', title='Phase 2: DNase降解未修饰C段',
           xlim=(5, 25), ylim=(0, 50))
    ax.legend()

    ax = axes[1, 1]
    cpp_pct = sol.y[3] / m.CPP_DNA_out0 * 100
    dcpp_pct = sol.y[5] / m.CPP_DNA_out0 * 100
    ax.plot(t, cpp_pct, 'r-', lw=2, label='CPP游离')
    ax.plot(t, dcpp_pct, 'k--', label='CPP已降解')
    ax.axhline(cpp_peak/2/m.CPP_DNA_out0*100, color='gray', ls=':', label='50% 峰值线')
    ax.set(xlabel='时间', ylabel='百分比(%)', title='Phase 3: 蛋白酶体降解CPP (t½=26s)',
           xlim=(10, 40))
    ax.legend()

    plt.tight_layout()
    plt.savefig('sim_2_CPP_delivery_PKPD.png', dpi=150)
    print("  📊 图已保存: sim_2_CPP_delivery_PKPD.png\n")
    plt.close()


# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "▓" * 65)
    print("  ELEANOR v4.0 模拟实验 — Part 1: KALTSIT + CPP递送")
    print("▓" * 65 + "\n")

    run_module_1A()
    run_module_1B()
    run_module_1C()
    run_module_2()

    print("▓" * 65)
    print("  Part 1 全部完成")
    print("▓" * 65)
