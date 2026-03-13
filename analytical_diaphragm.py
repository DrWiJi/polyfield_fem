# -*- coding: utf-8 -*-
"""
Аналитическая модель собственной частоты диафрагмы (мембрана) для валидации численной модели.

Мембрана: натянутая плёнка с закреплёнными краями, доминирует натяжение T (Н/м).

Соответствие с численной моделью (OpenCL):
  В ядре преднатяжение задаётся как pre_tension (Н/м). Добавляется удлинение связи:
    pre_elong = (pre_tension * edge_length) / k_soft,
  тогда в покое сила в связи F = k_eff * pre_elong ≈ k_soft * pre_elong = pre_tension * edge_length,
  т.е. натяжение по длине ребра T_eff = F / edge_length = pre_tension.
  Таким образом, численный pre_tension и аналитический T (натяжение на единицу длины) — одна и та же величина в Н/м; сопоставление корректно.
  Замечание: при нелинейной жёсткости k_eff в рабочей точке может отличаться от k_soft, тогда фактическое натяжение незначительно отличается от pre_tension.
"""
from __future__ import annotations

import numpy as np


def natural_frequency_membrane_rect(
    Lx: float,
    Ly: float,
    tension_per_unit_length: float,
    rho_surface: float,
    m: int = 1,
    n: int = 1,
) -> float:
    """
    Собственная частота f_mn прямоугольной мембраны с закреплёнными краями.

    Уравнение: T * (d²w/dx² + d²w/dy²) = ρ_s * d²w/dt².
    Решение: w ~ sin(m*π*x/Lx)*sin(n*π*y/Ly)*cos(ω*t),
    ω_mn = π * sqrt(T/ρ_s) * sqrt((m/Lx)² + (n/Ly)²).

    Параметры:
        Lx, Ly — размеры по x и y (м).
        tension_per_unit_length — натяжение T (Н/м).
        rho_surface — поверхностная плотность ρ_s (кг/м²) = ρ * h.
        m, n — номера мод (1,1 — первая мода).

    Возвращает f_mn в Гц.
    """
    if tension_per_unit_length <= 0 or rho_surface <= 0:
        return np.nan
    c_sq = tension_per_unit_length / (rho_surface + 1e-30)
    omega = np.pi * np.sqrt(c_sq * ((m / Lx) ** 2 + (n / Ly) ** 2))
    return float(omega / (2.0 * np.pi))


def analytical_natural_frequencies(
    width_m: float,
    height_m: float,
    thickness_m: float,
    density_kg_m3: float,
    E_parallel_pa: float,
    poisson: float,
    pre_tension_N_per_m: float,
) -> dict[str, float]:
    """
    Собственная частота мембраны (мода 1,1).

    Возвращает словарь с ключом membrane_f11_Hz.
    """
    rho_s = density_kg_m3 * thickness_m
    f_mem = natural_frequency_membrane_rect(
        width_m, height_m,
        pre_tension_N_per_m, rho_s,
        m=1, n=1,
    )
    return {"membrane_f11_Hz": f_mem}
