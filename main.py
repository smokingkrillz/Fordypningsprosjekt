import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dev.load_usm import USMLoader
from dev.plot_usm import USMPlotter




if __name__ == "__main__":
    loader = USMLoader()
    runs, gas = loader.load()
    run2 = runs["Vega 2st stage run 2"]
    run2 = runs["Vega 2st stage run 2"]
    print(run2[["q_vol", "zero_flow"]].head(20))
    print(run2["q_vol"].describe())
    loader.summary()

    plotter = USMPlotter(runs)
    plotter.plot_all()

