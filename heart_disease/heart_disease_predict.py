import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

df = pd.read_csv("datasets_33180_43520_heart.csv")
print(df.head)
