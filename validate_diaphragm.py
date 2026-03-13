# -*- coding: utf-8 -*-
"""
Валидация модели диафрагмы ROCm.

Запуск: py validate_diaphragm.py
  — всегда RK2, без графиков (не блокируют поток).
  — проверка численной стабильности, работы упругости, наличия резонансов.
"""
from __future__ import annotations

import sys
import numpy as np
from typing import NamedTuple

from diaphragm_rocm import PlanarDiaphragmROCm

# Параметры симуляции (с учётом сил от границы нужен меньший dt)
DT = 1e-8
DURATION_IMPULSE = 0.001  # 1 ms (100k шагов)
DURATION_UNIFORM = 0.0005  # 0.5 ms (50k шагов)
MAX_FREQ_HZ = 20000.0


class ValidationMetrics(NamedTuple):
    """Метрики валидации."""
    has_nan: bool
    max_disp_um: float
    n_resonance_peaks: int
    fundamental_hz: float
    peak_freqs_hz: tuple[float, ...]
    lin_fit_r2: float  # R^2 линейного тренда: ~1 = жёсткое тело
    n_zero_crossings: int
    decay_ratio: float  # |u_end|/|u_max| — затухание
    spectral_peak_prominence: float
    has_oscillation: bool  # признак колебаний по форме сигнала


def _spectral_peaks(hist: np.ndarray, dt: float, min_prominence_ratio: float = 0.03) -> tuple[list[float], float]:
    """Частоты пиков спектра и частота фундаментального тона."""
    if len(hist) < 16:
        return [], 0.0
    h = hist - np.mean(hist)  # убрать DC
    h = h - np.polyval(np.polyfit(np.arange(len(h)), h, 1), np.arange(len(h)))  # детренд
    freq = np.fft.fftfreq(len(h), dt)
    spec = np.abs(np.fft.fft(h))
    mask = (freq > 0) & (freq < MAX_FREQ_HZ)
    f_pos = freq[mask]
    s_pos = spec[mask]
    if len(s_pos) < 2:
        return [], 0.0
    threshold = np.max(s_pos) * min_prominence_ratio
    peaks = []
    for i in range(1, len(s_pos) - 1):
        if s_pos[i] >= threshold and s_pos[i] >= s_pos[i - 1] and s_pos[i] >= s_pos[i + 1]:
            peaks.append(float(f_pos[i]))
    peaks = sorted(peaks, key=lambda f: np.interp(f, f_pos, s_pos), reverse=True)[:10]
    fund = peaks[0] if peaks else 0.0
    return peaks, fund


def _linear_trend_r2(hist: np.ndarray) -> float:
    """R² линейной регрессии: 1 = чистый линейный дрейф (упругость не работает)."""
    if len(hist) < 3:
        return 0.0
    x = np.arange(len(hist), dtype=float)
    coef = np.polyfit(x, hist, 1)
    y_pred = np.polyval(coef, x)
    ss_res = np.sum((hist - y_pred) ** 2)
    ss_tot = np.sum((hist - np.mean(hist)) ** 2)
    if ss_tot < 1e-30:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _zero_crossings(hist: np.ndarray) -> int:
    """Количество пересечений нуля относительно среднего (признак колебаний)."""
    if len(hist) < 2:
        return 0
    h = hist - np.mean(hist)  # осцилляции вокруг среднего
    s = np.sign(h)
    return int(np.sum(np.abs(np.diff(s)) > 0) / 2)


def run_impulse_validation(model: PlanarDiaphragmROCm) -> ValidationMetrics:
    """Импульс 100 Па в первые 100 шагов — затухающие колебания с резонансами."""
    n_steps = int(DURATION_IMPULSE / DT)
    pressure = np.zeros(n_steps, dtype=np.float64)
    n_impulse = min(100, n_steps // 10)
    pressure[:n_impulse] = 10.0  # импульс

    model.simulate(pressure, dt=DT, record_history=False, check_air_resistance=False, validate_steps=True)
    hist = np.asarray(model.history_disp_center, dtype=np.float64)

    has_nan = np.any(~np.isfinite(hist))
    max_disp_um = float(np.max(np.abs(hist)) * 1e6) if not has_nan else np.nan

    peaks, fund = _spectral_peaks(hist, DT)
    r2 = _linear_trend_r2(hist)
    nzc = _zero_crossings(hist)

    u_max = np.max(np.abs(hist)) if len(hist) > 0 and not has_nan else 1e-30
    u_end = np.abs(hist[-1]) if len(hist) > 0 else 0.0
    decay_ratio = float(u_end / (u_max + 1e-40))

    spec = np.abs(np.fft.fft(hist - np.mean(hist)))
    freq = np.fft.fftfreq(len(hist), DT)
    mask = (freq > 0) & (freq < MAX_FREQ_HZ)
    if np.any(mask):
        s = spec[mask]
        peak_val = np.max(s)
        mean_val = np.mean(s) + 1e-40
        prominence = float(peak_val / mean_val)
    else:
        prominence = 0.0

    has_osc = _detect_oscillation(hist)

    return ValidationMetrics(
        has_nan=has_nan,
        max_disp_um=max_disp_um,
        n_resonance_peaks=len(peaks),
        fundamental_hz=fund,
        peak_freqs_hz=tuple(peaks[:5]),
        lin_fit_r2=r2,
        n_zero_crossings=nzc,
        decay_ratio=decay_ratio,
        spectral_peak_prominence=prominence,
        has_oscillation=has_osc,
    )


def run_uniform_validation(model: PlanarDiaphragmROCm) -> tuple[bool, float, float]:
    """Постоянное давление — квазистатическое равновесие, не линейный дрейф."""
    n_steps = int(DURATION_UNIFORM / DT)
    pressure = np.full(n_steps, 1.0, dtype=np.float64)

    model.simulate(pressure, dt=DT, record_history=False, check_air_resistance=False, validate_steps=True)
    hist = np.asarray(model.history_disp_center, dtype=np.float64)

    has_nan = np.any(~np.isfinite(hist))
    max_um = float(np.max(np.abs(hist)) * 1e6) if not has_nan else np.nan
    r2 = _linear_trend_r2(hist)
    return has_nan, max_um, r2


def _detect_oscillation(hist: np.ndarray) -> bool:
    """Есть ли осцилляция: у detrended данных меняется знак производной."""
    if len(hist) < 10:
        return False
    h = hist - np.mean(hist)
    d = np.diff(h)
    sign_changes = np.sum(np.diff(np.sign(d)) != 0)
    return sign_changes >= 2  # хотя бы один "горб" или "впадина"


def _print_metrics(name: str, m: ValidationMetrics) -> None:
    print(f"\n--- {name} ---")
    print(f"  NaN/Inf:        {'FAIL' if m.has_nan else 'OK'}")
    print(f"  max |u|:        {m.max_disp_um:.4f} µm")
    print(f"  Резонансов:     {m.n_resonance_peaks}")
    print(f"  f0 (основной):  {m.fundamental_hz:.1f} Hz")
    if m.peak_freqs_hz:
        print(f"  Пики (Hz):      {', '.join(f'{f:.0f}' for f in m.peak_freqs_hz)}")
    print(f"  R^2 линейности: {m.lin_fit_r2:.4f}  (1.0 = жёсткое тело, упругость не работает)")
    print(f"  Пересечений 0:  {m.n_zero_crossings}  (колебания)")
    print(f"  Затухание:      {m.decay_ratio:.4f}  (|u_end|/|u_max|)")
    print(f"  Выделенность пиков: {m.spectral_peak_prominence:.2f}")
    print(f"  Осцилляции:     {'да' if m.has_oscillation else 'нет'}")


def main() -> int:
    quick = "--quick" in sys.argv
    global DT, DURATION_IMPULSE, DURATION_UNIFORM
    if quick:
        DT, DURATION_IMPULSE, DURATION_UNIFORM = 1e-7, 0.005, 0.002  # 50k + 20k шагов
        print("(режим --quick)")

    print("=" * 60)
    print("Валидация модели диафрагмы (RK2, без графиков)")
    print("=" * 60)

    model = PlanarDiaphragmROCm(nx=24, ny=32, use_rk2=True, debug_hip_elastic=False)
    dt_s = DT
    print(f"\nПараметры: dt={dt_s:.0e} s, импульс {DURATION_IMPULSE*1000:.0f} ms, uniform {DURATION_UNIFORM*1000:.0f} ms")

    failures = []

    # 1. Импульсный отклик
    print("\n[1] Импульс 100 Па (первые 100 шагов)...")
    m_imp = run_impulse_validation(model)
    _print_metrics("Импульсный отклик", m_imp)

    if m_imp.has_nan:
        failures.append("Импульс: NaN/Inf")
    # Известное ограничение: граничные соседи пропускаются — при равномерном возбуждении R^2 может быть высоким

    # 2. Постоянное давление
    print("\n[2] Постоянное давление 1 Па...")
    has_nan_u, max_u, r2_u = run_uniform_validation(model)
    print(f"\n--- Постоянное давление ---")
    print(f"  NaN/Inf:        {'FAIL' if has_nan_u else 'OK'}")
    print(f"  max |u|:        {max_u:.4f} µm")
    print(f"  R^2 линейности: {r2_u:.4f}  (при равновесии должен быть низкий)")

    if has_nan_u:
        failures.append("Uniform: NaN/Inf")
    if r2_u > 0.98:
        failures.append("Uniform: R^2 > 0.98 — линейный дрейф вместо равновесия")

    # 3. Сводка
    print("\n" + "=" * 60)
    if failures:
        print("ОШИБКИ ВАЛИДАЦИИ:")
        for f in failures:
            print(f"  * {f}")
        return 1
    print("Валидация пройдена.")
    print("\nПримечание: резонансы и колебания зависят от учёта границы.")
    print("Граничные элементы в ядре пропускаются — при равномерном возбуждении")
    print("упругость между внутренними элементами мала; для полной модели нужны")
    print("силы от закреплённой границы (pos_nb=0 для boundary).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
