"""Invariant mass for dimuon samples (matches CMS education notebook formula)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def invariant_mass_dimuon_pt_eta_phi(df: pd.DataFrame) -> pd.Series:
    """
    M = sqrt(2 * pt1 * pt2 * (cosh(eta1 - eta2) - cos(phi1 - phi2)))

    Same as `Calculate-invariant-mass.ipynb` for Zmumu_Run2011A.csv.
    """
    pt1 = df["pt1"].astype(float)
    pt2 = df["pt2"].astype(float)
    eta1 = df["eta1"].astype(float)
    eta2 = df["eta2"].astype(float)
    phi1 = df["phi1"].astype(float)
    phi2 = df["phi2"].astype(float)
    inner = np.cosh(eta1 - eta2) - np.cos(phi1 - phi2)
    inner = np.clip(inner, 0.0, None)
    return np.sqrt(2 * pt1 * pt2 * inner)
