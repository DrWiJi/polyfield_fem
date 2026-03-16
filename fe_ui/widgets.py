# -*- coding: utf-8 -*-
"""
Shared widgets for FE UI. ScientificDoubleSpinBox displays values in exponential form.
"""

from __future__ import annotations

import re

from PySide6.QtCore import QLocale
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QDoubleSpinBox


class ScientificDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that displays and accepts values in scientific notation (e.g. 1.23e-4)."""

    def textFromValue(self, value: float) -> str:
        prec = self.decimals()
        # Use 'e' format: 1.23e-04
        return QLocale().toString(value, "e", prec)

    def valueFromText(self, text: str) -> float:
        s = str(text).strip()
        prefix, suffix = self.prefix(), self.suffix()
        if prefix and s.startswith(prefix):
            s = s[len(prefix) :].strip()
        if suffix and s.endswith(suffix):
            s = s[: -len(suffix)].strip()
        if not s:
            return self.minimum()
        try:
            return float(
                s.replace(QLocale().decimalPoint(), ".")
                .replace(QLocale().groupSeparator(), "")
            )
        except ValueError:
            return self.minimum()

    def validate(self, text: str, pos: int) -> tuple[QValidator.State, str, int]:
        # Accept scientific notation: 1.23e-4, 1e6, .5e2, etc.
        s = str(text).strip()
        prefix, suffix = self.prefix(), self.suffix()
        if prefix and s.startswith(prefix):
            s = s[len(prefix) :].strip()
        if suffix and s.endswith(suffix):
            s = s[: -len(suffix)].strip()
        if not s:
            return QValidator.State.Intermediate, text, pos
        # Allow incomplete input during typing (e.g. "1.23e", "1e-")
        normalized = s.replace(QLocale().decimalPoint(), ".").replace(
            QLocale().groupSeparator(), ""
        )
        try:
            float(normalized)
            return QValidator.State.Acceptable, text, pos
        except ValueError:
            # Check if it could be a prefix of valid input
            if s in ("-", "+", ".", "e", "E") or s.endswith(
                ("e", "E", "e-", "e+", "E-", "E+")
            ):
                return QValidator.State.Intermediate, text, pos
            # Allow "1.23e" or "1e" (incomplete exponent)
            if re.match(r"^[+-]?\d*\.?\d*[eE][+-]?\d*$", normalized):
                return QValidator.State.Intermediate, text, pos
            if re.match(r"^[+-]?\d*\.?\d*$", normalized):
                return QValidator.State.Intermediate, text, pos
            return QValidator.State.Invalid, text, pos
