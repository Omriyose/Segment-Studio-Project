from sklearn.datasets import load_iris
import pandas as pd

cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
df = pd.DataFrame(load_iris().data, columns=cols)
df.to_csv("iris_no_label.csv", index=False)