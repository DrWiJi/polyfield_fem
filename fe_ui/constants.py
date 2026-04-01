# -*- coding: utf-8 -*-
"""Constants for FE UI. No external dependencies."""

PROJECT_EXT = ".fe_project"
LIBRARY_EXT = ".fe_lib"

ROLES = ("solid", "membrane", "sensor")
FIXED_EDGE_OPTIONS = ("none", "FIXED_EDGE", "FIXED_ALL")
FORCE_SHAPES = ("impulse", "uniform", "sine", "square", "chirp", "white_noise")
EXCITATION_MODES = (
    "external",
    "external_full_override",
    "second_order_boundary_full_override",
    "external_velocity_override",
)
