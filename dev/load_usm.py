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

FILE = "input/Vega 2st stage low flow.xlsx"
HEADER_ROW = 6          # 0-based row index with the column names (header=None)
DATA_ROW = 8           # 0-based row index of the first data row

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


class USMLoader:
    """Reads the USM Excel export and builds one DataFrame per run sheet.

    Usage:
        loader = USMLoader("input/Vega 2st stage low flow.xlsx")
        runs, gas = loader.load()
    After load(), the results are also kept on the object as loader.runs and loader.gas.
    """

    def __init__(self, path=FILE):
        self.path = path
        self.runs = {}      # sheet name -> DataFrame, filled by load()
        self.gas = None     # gas-quality DataFrame, filled by load()

    @staticmethod
    def is_datetime_value(x):
        """True if a single cell value is a date/time object."""
        return isinstance(x, datetime.datetime)

    @staticmethod
    def to_datetime(col):
        """Convert one timestamp column from the Excel sheet into proper pandas datetimes.
        The column has dtype "object", but some cells are datetimes or
        when the column also contains text or blanks.
        The result is rounded to whole seconds, because Excel's
        floating-point storage gives values like 12:00:00.000001 that would
        otherwise fail to match when joining sheets on time.
        """
        is_dt = col.map(USMLoader.is_datetime_value)
        only_dates = col.where(is_dt)
        return pd.to_datetime(only_dates, errors="coerce").dt.round("s")

    def read_sheet(self, sheet, rename):
        raw = pd.read_excel(self.path, sheet_name=sheet, header=None)
        # header row in location 7
        names = raw.iloc[HEADER_ROW].tolist()
        df = raw.iloc[DATA_ROW:].copy()

        time_columns = ["t_start", "t_end"]

        value_columns = []
        for original_name in names[2:]: 
            if original_name in rename:
                new_name = rename[original_name]
            else:
                new_name = original_name
            value_columns.append(new_name)

        # Put the two lists together and use them as the column names
        df.columns = time_columns + value_columns
        print(value_columns)
        df["t_start"] = self.to_datetime(df["t_start"])
        df = df.dropna(subset=["t_start"]).drop(columns="t_end")

        
        named_columns = []
        for c in df.columns:
            if not pd.isna(c):
                named_columns.append(c)
        df = df[named_columns]
        df = df.set_index("t_start").sort_index()             # index first, so it is not coerced to numbers
        return df.apply(pd.to_numeric, errors="coerce")

    @staticmethod
    def add_indicators(df):
        # cross-path VOS deviation: path minus mean of the other paths
        for p in PATHS:
            others = [q for q in PATHS if q != p]
            df[f"dev_{p}"] = df[f"vos_{p}"] - df[[f"vos_{q}" for q in others]].mean(axis=1)
        # GC discrepancy: calculated from GC minus measured
        df["gc_disc"] = df["vos_gc"] - df["vos_avg"]
        return df

    def load(self):
        xl = pd.ExcelFile(self.path)
        run_sheets = [s for s in xl.sheet_names if "run" in s.lower()]
        gas_sheet = [s for s in xl.sheet_names if "gas" in s.lower()][0]
        self.gas = self.read_sheet(gas_sheet, GAS_RENAME)
        self.runs = {}
        for s in run_sheets:
            df = self.read_sheet(s, RENAME)
            df = df.join(self.gas.drop(columns="vos_gc"), how="left")   # composition aligned on t_start
            df["zero_flow"] = df["q_vol"].fillna(0).eq(0)               # flag, don't drop yet
            self.runs[s] = self.add_indicators(df)
        return self.runs, self.gas

    def summary(self):
        for k, v in self.runs.items():
            print(k, v.shape, "zero-flow rows:", int(v.zero_flow.sum()), "NaNs:", int(v.isna().sum().sum()))


def load(path=FILE):
    """Shortcut so existing code (plot_usm.py) can keep calling load_usm.load()."""
    return USMLoader(path).load()
