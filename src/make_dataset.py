"""Snapshot the California housing dataset to data/california.csv (committed for reproducible, offline training)."""
from sklearn.datasets import fetch_california_housing



housing = fetch_california_housing(as_frame=True)
dataframe = housing.frame
dataframe.to_csv("data/california.csv",index=False)

print(dataframe.shape)
print(dataframe.head(-3))


