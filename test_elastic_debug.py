# -*- coding: utf-8 -*-
"""
Скрипт для проверки расчёта и применения упругой силы.
Запуск: python test_elastic_debug.py

При uniform pressure мембрана смещается как жёсткое тело — d_vec_z=0, F_el=0.
Давление только на центр создаёт градиент смещения → упругая сила в z.
"""
import numpy as np
from diaphragm_rocm import PlanarDiaphragmROCm

def main():
    model = PlanarDiaphragmROCm(nx=24, ny=32, debug_hip_elastic=True)
    dt = 1e-8  # меньший dt для стабильности при локальном давлении
    n_steps = 200
    # Давление только на центральный элемент — создаёт градиент uz и упругую силу
    pressure = np.zeros((n_steps, model.n_elements), dtype=np.float64)
    pressure[0, model.center_idx] = 1.0  # 1 Па на центр

    print("=== Валидация упругой силы (давление на центр) ===\n")
    hist = model.simulate(
        pressure, dt=dt, record_history=True,
        check_air_resistance=False,
        validate_steps=True,
    )

    print("\n=== Результат ===")
    if np.any(~np.isfinite(hist)):
        print("NaN/Inf в истории (останов на первом) — уменьшите dt")
    else:
        print(f"max |u_center| = {np.max(np.abs(hist))*1e6:.4f} µm")
        ok = np.max(np.abs(hist)) > 1e-9
        print("OK: упругая сила применяется" if ok else "ОШИБКА: смещение ~0 — упругость не работает")

if __name__ == "__main__":
    main()
