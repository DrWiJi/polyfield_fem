# -*- coding: utf-8 -*-
"""
Запуск валидации численной модели диафрагмы (OpenCL) против аналитической (мембрана).
Результаты выводятся в консоль и сохраняются в validation_report.txt (UTF-8).
"""
from __future__ import annotations

def run_validation(save_report: bool = True):
    from diaphragm_opencl import PlanarDiaphragmOpenCL, validate_natural_frequencies

    model = PlanarDiaphragmOpenCL(nx=24, ny=32, pre_tension_N_per_m=10.0)
    result = validate_natural_frequencies(
        model, dt=2e-7, duration=0.005, impulse_velocity_z=0.01
    )

    if save_report and result:
        report_path = "validation_report.txt"
        err = result.get("err_membrane_pct", float("nan"))
        err_str = f"{err:+.1f} %" if err == err else "—"  # NaN != NaN
        lines = [
            "Валидация: численная модель (OpenCL) vs аналитическая (мембрана)",
            "=" * 60,
            "",
            "--- Собственная частота (мембрана, мода 1,1) ---",
            f"  Аналитика:  f = {result.get('membrane_f11_Hz', 0):.2f} Hz",
            f"  Численная:  f = {result.get('numerical_f11_Hz', 0):.2f} Hz",
            f"  Ошибка:     {err_str}",
            "",
            "--- Критерии корректности ---",
            f"  max |uz|: {result.get('max_uz_um', 0):.2f} µm",
            f"  Выделенность пика: {result.get('peak_prominence', 0):.2f}",
            "",
        ]
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Отчёт сохранён: {report_path}")

    return result


if __name__ == "__main__":
    run_validation(save_report=True)
