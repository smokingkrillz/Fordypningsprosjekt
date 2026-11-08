import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dev.load_usm import USMLoader
from dev.plot_usm import USMPlotter
from dev.prepare import Prepare
from dev.stability import StabilityAnalysis
from dev.forecast_data import ForecastDataset
from dev.baselines import BaselineEvaluation
from dev.models import ModelEvaluation

src, out_dir = "Vega_2st_stage_run_1_prepared.csv", "results"

if __name__ == "__main__":
    loader = USMLoader()
    meter_data, gas_data = loader.load()
    loader.summary()
    for name, df in meter_data.items():
        p = Prepare(df, name=name)
        p.prepare().report().save(out_dir)

        if not p.df["valid"].any():
            print(f"{name}: no valid rows, skipping stability analysis")
            continue
        #StabilityAnalysis(p).report().save(out_dir)
         # Make the dataset (this is what XGBoost will train on later)
        ds = ForecastDataset(p, target="dev_p1", horizon=5, lookback=10).check()
        splits = ds.split(train=0.6, val=0.2)  
        # # step 2: every indicator, horizons 1/5/15 min, all baselines
        ev = BaselineEvaluation(p, horizons=(1, 5, 15), lookback=10).run().report().save("results")
        me = ModelEvaluation(ev).run().report().save("results")