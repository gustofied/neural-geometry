import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dir = Path(__file__).resolve().parent
data_path = dir / "data"/ "clean_weather.csv"
print(dir)

data = pd.read_csv(data_path)
data= data.fillna(data.mean(numeric_only=True))

print(data.head())
print(data.shape)

data.plot.scatter("tmax", "tmax_tomorrow")
plt.plot([30, 120], [30, 120], "green")
# plt.show()

lr = LinearRegression()
lr.fit(data[["tmax"]], data["tmax_tomorrow"])

fig, ax = plt.subplots()
data.plot.scatter("tmax", "tmax_tomorrow", ax=ax)
ax.plot(data["tmax"], lr.predict(data[["tmax"]]), "red")
print(f"Wegiht: {lr.coef_[0]:.2f}")
print(f"Bias {lr.intercept_:.2f}")
plt.show()

