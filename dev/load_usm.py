"""Loader for the Vega 2nd stage USM export (one sheet per run + one gas-quality sheet).

Sheet layout (Excel rows, 1-based):
    pandas rows 0-5  metadata (start/end/delta) + blanks   -> skipped
    pandas row 6     tag identifiers                       -> skipped
    pandas row 7     column names (col B is interval END)
    pandas row 8     units                                  -> skipped
    pandas row 9+    data. Col A = interval START, col B = interval END, col C+ = values
"""
import datetime

import numpy as np
import pandas as pd

FILE = "Vega 2st stage low flow.xlsx"
HEADER_ROW = 7          # 0-based row index with the column names (header=None)
DATA_ROW = 9            # 0-based row index of the first data row

RENAME = {
    "Volume flow rate": "q_vol",
    "Mass flow rate": "q_mass",
    "Reference volume flow rate": "q_ref",
    "Pressure used": "p",
    "Temperature used": "T",
    "VOS calculated from gas composition": "vos_gc",
    "Average velocity of sound": "vos_avg",
    "Measured VOS path 01": "vos_p1",
    "Measured VOS path 02": "vos_p2",
    "Measured VOS path 03": "vos_p3",
    "Flow velocity": "v",
    "Measured flow velocity path 01": "v_p1",
    "Measured flow velocity path 02": "v_p2",
    "Measured flow velocity path 03": "v_p3",
    "S/N ratio used signal path 01": "snr_p1",
    "S/N ratio used signal path 02": "snr_p2",
    "S/N ratio used signal path 03": "snr_p3",
}
GAS_RENAME = {
    "VOS calculated from gas composition": "vos_gc",
    "Reference density used": "rho_ref",
    "Mole weight used": "mw",
    "Used nitrogen (N2)": "N2", "Used carbon dioxide (CO2)": "CO2",
    "Used methane (C1)": "C1", "Used ethane (C2)": "C2", "Used propane (C3)": "C3",
    "Used i-butane (iC4)": "iC4", "Used n-butane (nC4)": "nC4",
    "Used i-pentane (iC5)": "iC5", "Used n-pentane (nC5)": "nC5",
    "Used hexane plus (C6+)": "C6plus",
}
PATHS = ["p1", "p2", "p3"]



def to_datetime(col):
    """Convert one timestamp column from the Excel sheet into proper pandas datetimes.

    1. The column has dtype "object" (mixed types), but some cells are datetimes.
       Happens when the column also contains text or blanks, e.g. a "Total" row
       at the bottom. We keep the datetime cells and turn everything else into NaT.


    In every case the result is rounded to whole seconds, because Excel's
    floating-point storage gives values like 12:00:00.000001 that would
    otherwise fail to match when joining sheets on time.
    """
    is_datetime_value =isinstance(col, datetime.datetime)
    is_dt = col.map(is_datetime_value)          # True/False for each cell
    only_dates = col.where(is_dt)           # non-datetime cells -> NaN
     # True if a single cell value is a date/time object (pandas Timestamp is a subclass of datetime)
    return pd.to_datetime(only_dates, errors="coerce").dt.round("s")

   

def read_sheet(path, sheet, rename):
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    names = raw.iloc[HEADER_ROW].tolist()
    df = raw.iloc[DATA_ROW:].copy()
    df.columns = ["t_start", "t_end"] + [rename.get(n, n) for n in names[2:]]
    df["t_start"] = to_datetime(df["t_start"])
    df = df.dropna(subset=["t_start"]).drop(columns="t_end")
    df = df[[c for c in df.columns if not pd.isna(c)]]
    df = df.set_index("t_start").sort_index()             # index first, so it is not coerced to numbers
    return df.apply(pd.to_numeric, errors="coerce")


def add_indicators(df):
    # cross-path VOS deviation: path minus mean of the other paths
    for p in PATHS:
        others = [q for q in PATHS if q != p]
        df[f"dev_{p}"] = df[f"vos_{p}"] - df[[f"vos_{q}"for q in others]].mean(axis=1)
    # GC discrepancy: calculated from GC minus measured
    df["gc_disc"] = df["vos_gc"] - df["vos_avg"]
    return df


def load(path=FILE):
    xl = pd.ExcelFile(path)
    run_sheets = [s for s in xl.sheet_names if "run" in s.lower()]
    gas = read_sheet(path, [s for s in xl.sheet_names if "gas" in s.lower()][0], GAS_RENAME)
    runs = {}
    for s in run_sheets:
        df = read_sheet(path, s, RENAME)
        df = df.join(gas.drop(columns="vos_gc"), how="left")   # composition aligned on t_start
        df["zero_flow"] = df["q_vol"].fillna(0).eq(0)          # flag, don't drop yet
        runs[s] = add_indicators(df)
    return runs, gas


if __name__ == "__main__":
    runs, gas = load()
    for k, v in runs.items():
        print(k, v.shape, "zero-flow rows:", int(v.zero_flow.sum()), "NaNs:", int(v.isna().sum().sum()))
