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
        meter_data, gas_data = loader.load()
    After load(), the results are also kept on the object as loader.runs and loader.gas.
    """
    def __init__(self, path=FILE):
        self.path = path
        self.meter_data = {}      # sheet name -> DataFrame, filled by load()
        self.gas_data = None     # gas-quality DataFrame, filled by load()

    @staticmethod
    def is_datetime_value(x) -> bool:
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
        """ 
        Takes one excel sheet, renames columns.

        """
        raw = pd.read_excel(self.path, sheet_name=sheet, header=None)

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
        #convert to real time
        df["t_start"] = self.to_datetime(df["t_start"])
        #drop missing t values and not need end time
        df = df.dropna(subset=["t_start"]).drop(columns="t_end")

        #  Keep only the columns that have a name
        named_columns = []
        for column_names in df.columns:
            if not pd.isna(column_names):
                named_columns.append(column_names)
        
        # index first, so it is not coerced to numbers
        df = df[named_columns]
        df = df.set_index("t_start").sort_index()  
        # making every value a number, coerce makes bad data to nan   
        return df.apply(pd.to_numeric, errors="coerce")

    @staticmethod
    def add_indicators(df):
        """cross-path VOS deviation: path minus mean of the other paths
        
            input parameters:
            df : dtype        
        """
        for path in PATHS:         
            # (a) find the names of the OTHER two paths
            other_paths = []
            for candidate in PATHS:
                if candidate != path:
                    other_paths.append(candidate)

            # (b) this path's velocity of sound: f"vos_{path}" becomes "vos_p1"
            this_path_vos = df[f"vos_{path}"]   

            # (c) the other two paths' columns, and their average for each minute
            other_column_names = []
            for other in other_paths:
                other_column_names.append(f"vos_{other}")
            other_paths_vos = df[other_column_names]
            # axis=1 means: average across the row
            average_of_others = other_paths_vos.mean(axis=1)   

            # (d) the new column: this path minus the average of the others
            df[f"dev_{path}"] = this_path_vos - average_of_others

        # GC discrepancy: velocity of sound from the gas composition minus the measured one
        vos_from_gas = df["vos_gc"]
        vos_measured = df["vos_avg"]
        df["gc_disc"] = vos_from_gas - vos_measured
        return df

    def load(self):
        """ Finds the right sheets, uses the other functions
        adds gass composition
        """
        excel_file = pd.ExcelFile(self.path)
        all_sheet_names = excel_file.sheet_names
        # Sheets with "run" in the name are the meter data
        meter_data_sheets = []
        for sheet_name in all_sheet_names:
            if "run" in sheet_name.lower():
                meter_data_sheets.append(sheet_name)

        # The sheet with "gas" in the name is the gas composition (take the first one)
        gas_sheets = []
        for sheet_name in all_sheet_names:
            if "gas" in sheet_name.lower():
                gas_sheets.append(sheet_name)
        gas_sheet = gas_sheets[0]


        self.gas = self.read_sheet(gas_sheet, GAS_RENAME)

        # a dictionary: a set of labelled tables
        self.meter_data = {}
        for sheet_name in meter_data_sheets:
            df = self.read_sheet(sheet_name, RENAME)


            # Attach the gas composition to each minute
            gas_without_vos = self.gas.drop(columns="vos_gc")
            df = df.join(gas_without_vos, how="left")
            # First draft of the zero-flow flag: True where flow is 0 (or missing)
            flow_with_gaps_filled = df["q_vol"].fillna(0)
            df["zero_flow"] = flow_with_gaps_filled.eq(0)
            self.meter_data[sheet_name] = self.add_indicators(df)

        return self.meter_data, self.gas
    

    def summary(self):
        """Print the size of every table."""
        for name, df in self.meter_data.items():
            number_of_zero_flow = int(df["zero_flow"].sum())
            number_of_missing = int(df.isna().sum().sum())
            print(name, df.shape, "zero-flow rows:", number_of_zero_flow, "NaNs:", number_of_missing)


def load(path=FILE):
    """Shortcut so existing code (plot_usm.py) can keep calling load_usm.load()."""
    return USMLoader(path).load()
