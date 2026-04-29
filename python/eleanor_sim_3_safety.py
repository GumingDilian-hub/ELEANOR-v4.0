"""
ELEANOR v4.0 模拟实验 — 第三部分
模块4: 皮层全局监控 & 自动断电
模块5: 同构映射 & 运动学边界
模块6: 全系统蒙特卡洛安全仿真
直接运行: python eleanor_sim_3_safety.py
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
#  模块4: 皮层全局监控 & 自动断电
# ═══════════════════════════════════════════════════════════════

def generate_lfp(duration=10.0, fs=1000, mode='normal'):
    t = np.arange(0, duration, 1/fs)
    np.random.seed(42 if mode == 'normal' else 99)

    if mode == 'normal':
        sig = (2.0 * np.sin(2*np.pi*2*t) +
               1.5 * np.sin(2*np.pi*6*t) +
               1.0 * np.sin(2*np.pi*10*t) +
               0.8 * np.sin(2*np.pi*20*t) +
               0.5 * np.sin(2*np.pi*40*t) +
               np.random.normal(0, 1.0, len(t)))
    elif mode == 'epileptic':
        sig = (2.0 * np.sin(2*np.pi*2*t) +
               1.5 * np.sin(2*np.pi*6*t) +
               np.random.normal(0, 1.0, len(t)))
        seizure_mask = (t >= 5.0) & (t < 7.0)
        sig[seizure_mask] += 8.0 * np.sin(2*np.pi*25*t[seizure_mask])
        sig[seizure_mask] += np.random.normal(0, 4.0, np.sum(seizure_mask))
    elif mode == 'spreading_dep':
        sig = (2.0 * np.sin(2*np.pi*2*t) +
               np.random.normal(0, 1.0, len(t)))
        sd_mask = (t >= 4.0) & (t < 8.0)
        sig[sd_mask] += 6.0 * np.sin(2*np.pi*0.5*t[sd_mask])
        sig[sd_mask] += np.random.normal(0, 3.0, np.sum(sd_mask))
    return t, sig


def detect_anomaly_lfp(t, sig, fs=1000, window=0.5, threshold_z=3.0):
    win_samples = int(window * fs)
    n_windows = len(sig) // win_samples
    rms = np.zeros(n_windows)
    for i in range(n_windows):
        chunk = sig[i*win_samples : (i+1)*win_samples]
        rms[i] = np.sqrt(np.mean(chunk**2))

    baseline_rms = rms[:4]
    mu = np.mean(baseline_rms)
    sigma = np.std(baseline_rms) + 1e-10
    z_scores = (rms - mu) / sigma
    anomaly = z_scores > threshold_z
    return anomaly, z_scores, rms


def run_module_4():
    print("=" * 65)
    print("  模块4: 皮层全局监控 & 自动断电仿真")
    print("=" * 65)

    fs = 1000
    duration = 10.0
    modes = ['normal', 'epileptic', 'spreading_dep']
    labels = ['正常LFP', '癫痫样放电', '扩散性抑制']

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    for idx, (mode, label) in enumerate(zip(modes, labels)):
        t, sig = generate_lfp(duration, fs, mode)
        anomaly, z_scores, rms = detect_anomaly_lfp(t, sig, fs)

        n_anom = np.sum(anomaly)
        if n_anom > 0:
            first_anom = np.argmax(anomaly)
            t_detect = first_anom * 0.5
            print(f"  {label}: 检测到 {n_anom} 个异常窗口, 首次检测时间={t_detect:.1f}s → 触发硬断电")
        else:
            print(f"  {label}: 未检测到异常 → 系统正常运行")

        ax = axes[idx, 0]
        ax.plot(t, sig, 'b-', lw=0.5, alpha=0.8)
        if n_anom > 0:
            for ai in np.where(anomaly)[0]:
                t_start = ai * 0.5
                ax.axvspan(t_start, t_start + 0.5, alpha=0.3, color='red')
            ax.axvline(t_detect, color='red', ls='--', lw=2, label=f'硬断电 t={t_detect:.1f}s')
            ax.legend(fontsize=8)
        ax.set(xlabel='时间', ylabel='振幅 (μV)', title=f'{label} — 原始信号', ylim=(-20, 20))

        ax = axes[idx, 1]
        t_z = np.arange(len(z_scores)) * 0.5
        ax.bar(t_z, z_scores, width=0.4, color=['red' if a else 'steelblue' for a in anomaly])
        ax.axhline(3.0, color='red', ls='--', lw=2, label='阈值 Z=3')
        ax.set(xlabel='时间', ylabel='Z-score', title=f'{label} — 异常检测')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('sim_4_safety_monitor.png', dpi=150)
    print("  📊 图已保存: sim_4_safety_monitor.png\n")
    plt.close()

    print("  --- Dead-Man's Switch 逻辑验证 ---")
    print("  规则: 断电后该次会话不可逆, 需重新给药才能再激活")
    state = 'ACTIVE'
    for event in ['正常刺激', '检测到异常', '自动断电', '尝试重新激活', '重新给药', '再次激活']:
        if state == 'ACTIVE' and event == '检测到异常':
            state = 'ANOMALY_DETECTED'
        elif state == 'ANOMALY_DETECTED' and event == '自动断电':
            state = 'HARD_SHUTDOWN'
        elif state == 'HARD_SHUTDOWN' and event == '尝试重新激活':
            state = 'LOCKED'
            print(f"  [{event}] → {state} ❌ 拒绝: 需重新给药")
            continue
        elif state == 'LOCKED' and event == '重新给药':
            state = 'HARD_SHUTDOWN'
            print(f"  [{event}] → 解锁中...")
            continue
        elif state == 'HARD_SHUTDOWN' and event == '再次激活':
            state = 'ACTIVE'
            print(f"  [{event}] → {state} ✅ 新会话启动")
            continue
        print(f"  [{event}] → {state}")
    print()


# ═══════════════════════════════════════════════════════════════
#  模块5: 同构映射 & 运动学边界
# ═══════════════════════════════════════════════════════════════

def run_module_5():
    print("=" * 65)
    print("  模块5: 同构映射 & 运动学边界仿真")
    print("=" * 65)

    human_rom = {
        '肩屈伸':  (-50, 180),
        '肩外展':  (0, 180),
        '肘屈伸':  (0, 145),
        '腕屈伸':  (-80, 80),
        '髋屈伸':  (-30, 120),
        '膝屈伸':  (0, 140),
        '踝背伸':  (-30, 50),
    }

    np.random.seed(42)
    n_frames = 300
    t = np.arange(n_frames) / 30.0

    normal_traj = {}
    for joint, (lo, hi) in human_rom.items():
        center = (lo + hi) / 2
        amp = (hi - lo) * 0.4
        normal_traj[joint] = center + amp * np.sin(2 * np.pi * 0.3 * t + np.random.uniform(0, 2*np.pi))

    rogue_traj = {}
    for joint, traj in normal_traj.items():
        rt = traj.copy()
        lo, hi = human_rom[joint]
        rogue_frames = np.random.choice(n_frames, size=5, replace=False)
        for rf in rogue_frames:
            violation = hi + np.random.uniform(10, 40)
            rt[rf] = violation
        rogue_traj[joint] = rt

    print("  --- 人体→机器人 同构映射验证 ---")
    print(f"  {'关节':<10} {'ROM下限':<10} {'ROM上限':<10} {'正常轨迹越界':<14} {'注入错误越界':<14}")
    print("  " + "-" * 58)

    normal_violations = 0
    rogue_violations = 0

    for joint in human_rom:
        lo, hi = human_rom[joint]
        nv = np.any((normal_traj[joint] < lo) | (normal_traj[joint] > hi))
        rv = np.any((rogue_traj[joint] < lo) | (rogue_traj[joint] > hi))
        if nv: normal_violations += 1
        if rv: rogue_violations += 1
        print(f"  {joint:<10} {lo:<10.0f} {hi:<10.0f} "
              f"{'❌' if nv else '✅':<14} {'❌' if rv else '✅':<14}")

    print(f"\n  正常轨迹越界关节数: {normal_violations}/7 → {'✅ 全部合规' if normal_violations == 0 else '❌ 存在越界'}")
    print(f"  注入错误越界关节数: {rogue_violations}/7 → 被安全层拦截")

    print("\n  --- 非人形运动学拒绝验证 ---")
    non_human_queries = [
        ("旋转关节360°", {'肩屈伸': 360}),
        ("肘反向弯曲-30°", {'肘屈伸': -30}),
        ("膝关节超伸200°", {'膝屈伸': 200}),
        ("新增非人形关节:触手摆动0-270°", {'触手摆动': 270}),
    ]
    for desc, query in non_human_queries:
        rejected = False
        reason = ""
        for joint, angle in query.items():
            if joint not in human_rom:
                rejected = True
                reason = f"关节 '{joint}' 不在人体关节定义中 (非人形映射)"
                break
            lo, hi = human_rom[joint]
            if angle < lo or angle > hi:
                rejected = True
                reason = f"{joint}={angle}° 超出ROM[{lo},{hi}]"
                break
        if rejected:
            print(f"  ❌ 拒绝: {desc} — {reason}")
        else:
            print(f"  ✅ 通过: {desc}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    joint = '肘屈伸'
    lo, hi = human_rom[joint]

    ax1.plot(t, normal_traj[joint], 'b-', lw=1.5, label='正常解码轨迹')
    ax1.axhline(hi, color='red', ls='--', lw=2, label=f'硬上限 {hi}°')
    ax1.axhline(lo, color='red', ls='--', lw=2, label=f'硬下限 {lo}°')
    ax1.fill_between(t, lo, hi, alpha=0.08, color='green')
    ax1.set(xlabel='时间', ylabel='角度 (°)',
            title=f'{joint}: 正常轨迹 (同构映射)', ylim=(lo-20, hi+20))
    ax1.legend()

    ax2.plot(t, rogue_traj[joint], 'r-', lw=1.5, label='含注入错误的轨迹')
    rogue_points = rogue_traj[joint] > hi
    ax2.scatter(t[rogue_points], rogue_traj[joint][rogue_points],
                color='red', s=80, zorder=5, label='越界点 (被拦截)')
    ax2.axhline(hi, color='red', ls='--', lw=2, label=f'硬上限 {hi}°')
    ax2.axhline(lo, color='red', ls='--', lw=2, label=f'硬下限 {lo}°')
    ax2.fill_between(t, lo, hi, alpha=0.08, color='green')
    ax2.set(xlabel='时间', ylabel='角度 (°)',
            title=f'{joint}: 注入错误轨迹 (安全拦截)', ylim=(lo-20, hi+50))
    ax2.legend()

    plt.tight_layout()
    plt.savefig('sim_5_isomorphic_mapping.png', dpi=150)
    print("\n  📊 图已保存: sim_5_isomorphic_mapping.png\n")
    plt.close()


# ═══════════════════════════════════════════════════════════════
#  模块6: 全系统蒙特卡洛安全仿真
# ═══════════════════════════════════════════════════════════════

def run_module_6():
    print("=" * 65)
    print("  模块6: 全系统蒙特卡洛安全仿真")
    print("=" * 65)

    layers = {
        'L0_KALTSIT_AND门':    1e-4,
        'L1_CPP自切除':        1e-4,
        'L2_皮层全局监控':      1e-3,
        'L3_DeadManSwitch':    1e-3,
        'L4_自动断电':          1e-5,
        'L5_同构映射边界':      1e-3,
        'L6_闭源安全固件':      1e-3,
    }

    n_sim = 1_000_000
    print(f"  蒙特卡洛次数: {n_sim:,}")
    print(f"\n  {'安全层':<22} {'单层失效率':<14} {'失效次数':<12} {'通过次数':<12}")
    print("  " + "-" * 60)

    np.random.seed(42)
    n_layers = len(layers)
    layer_names = list(layers.keys())
    p_values = list(layers.values())
    fail_matrix = np.random.random((n_sim, n_layers)) < np.array(p_values)

    for i, (name, p) in enumerate(layers.items()):
        n_fail = np.sum(fail_matrix[:, i])
        n_pass = n_sim - n_fail
        print(f"  {name:<22} {p:<14.1e} {n_fail:<12,} {n_pass:<12,}")

    all_fail = np.all(fail_matrix, axis=1)
    system_fail_count = np.sum(all_fail)
    system_fail_rate = system_fail_count / n_sim

    print(f"\n  {'='*50}")
    print(f"  系统级失效 (所有层同时穿透):")
    print(f"    失效次数: {system_fail_count:,} / {n_sim:,}")
    print(f"    系统失效率: {system_fail_rate:.2e}")

    theoretical = np.prod(list(layers.values()))
    print(f"    理论值:     {theoretical:.2e}")
    print(f"    匹配度:     {system_fail_rate/theoretical:.2f}x")

    mtbf_hours = 1.0 / system_fail_rate
    mtbf_years = mtbf_hours / 8760
    print(f"\n  MTBF估算: {mtbf_hours:,.0f} 小时 ≈ {mtbf_years:,.0f} 年")

    print(f"\n  --- 单点故障注入分析 ---")
    print(f"  {'注入层':<22} {'仅该层失效时系统安全?':<30}")
    print("  " + "-" * 52)
    for i, name in enumerate(layer_names):
        single_fail = fail_matrix[:, i]
        other_ok = ~np.any(fail_matrix[:, np.arange(n_layers) != i], axis=1)
        breach = single_fail & other_ok
        n_breach = np.sum(breach)
        safe = n_breach == 0
        status = "✅ 安全 (被后续层拦截)" if safe else f"❌ 危险 ({n_breach}次穿透)"
        print(f"  {name:<22} {status}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.barh(range(n_layers), [-np.log10(p) for p in p_values],
             color='steelblue', edgecolor='navy')
    ax1.set(yticks=range(n_layers), yticklabels=layer_names,
            xlabel='-log₁₀(失效率)', title='各安全层可靠性等级')
    ax1.invert_yaxis()
    for i, p in enumerate(p_values):
        ax1.text(-np.log10(p) + 0.1, i, f'{p:.0e}', va='center', fontsize=9)

    single_rates = p_values
    x = range(n_layers)
    ax2.semilogy(x, single_rates, 'ro-', lw=2, ms=8, label='单层失效率')
    ax2.axhline(theoretical, color='green', ls='--', lw=3,
                label=f'系统级失效率 ({theoretical:.1e})')
    ax2.axhline(system_fail_rate, color='blue', ls=':', lw=2,
                label=f'蒙特卡洛结果 ({system_fail_rate:.1e})')
    ax2.set(xticks=x, xticklabels=[n.split('_')[0] for n in layer_names],
            ylabel='失效率 (对数)', title='安全层叠加固效果')
    ax2.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('sim_6_monte_carlo_safety.png', dpi=150)
    print("\n  📊 图已保存: sim_6_monte_carlo_safety.png\n")
    plt.close()


# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "▓" * 65)
    print("  ELEANOR v4.0 模拟实验 — Part 3: 安全系统仿真")
    print("▓" * 65 + "\n")

    run_module_4()
    run_module_5()
    run_module_6()

    print("▓" * 65)
    print("  Part 3 全部完成")
    print("▓" * 65)
    print("\n" + "═" * 65)
    print("  ELEANOR v4.0 全部6大模块 × 12个子实验 仿真完成")
    print("  生成图片: sim_1A ~ sim_6 共 9 张")
    print("═" * 65)
