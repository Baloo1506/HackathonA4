import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("HRDataset_v14_enriched.csv")
df.head()
df.shape
df.columns
df.info()
df.describe()
df.isnull().sum()
