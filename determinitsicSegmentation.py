import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file = "data/20220418_EVLP818_converted.csv"

df = pd.read_csv(file, header=0, sep=',', parse_dates=True, index_col=0)

df.index = df.index - df.index[0]