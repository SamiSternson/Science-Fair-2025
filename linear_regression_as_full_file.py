# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
import math

# %%
df = pd.read_csv("compiled_data.csv")
df

# %%
from scipy import stats

ys = list(df["Angle of Repose"])
print(ys)

# %%
z_scored_ys = stats.zscore(ys)

# %%
xs = []
for i in range(1, len(list(df.iloc[1:,]))):
    xs.append([list(df.iloc[i - 1 :, i])[j] for j in range(0, 40, 10)])
# for i in range(len(xs)):
#     xs[i]=[val**0.25 for val in xs[i]]
zd_xs = [list(stats.zscore(x)) for x in xs]


# %%
new_skeleton = [[] for i in range(len(xs))]
for i, column in enumerate(zd_xs):
    print(column)
    for cell in column:
        for j in range(10):
            new_skeleton[i].append(cell)


# %%
zd_df = pd.DataFrame(
    {
        "Angle of Repose": z_scored_ys,
        "Length to Width Ratio": new_skeleton[0],
        "Coeff of Friction": new_skeleton[1],
        "Density": new_skeleton[2],
        "Volume": new_skeleton[3],
        "Length to Width Standard Deviation": new_skeleton[4],
        "Coeff of Friction Standard Deviation": new_skeleton[5],
    }
)
zd_df

# %%
plt.scatter(df["Angle of Repose"], zd_df["Angle of Repose"])
plt.show()

# %%
y = zd_df["Angle of Repose"]
X = zd_df.drop(["Angle of Repose"], axis=1).astype("float64")

# %%
from sklearn.linear_model import LassoCV, Lasso

avg_error = 0
model_scores = []
all_coeffs = []
for i in range(0, 10):
    X_train = X.drop([(i + j * 10) for j in range(4)])
    y_train = y.drop([(i + j * 10) for j in range(4)])
    X_test = X.drop(X_train.index)
    y_test = y.drop(y_train.index)
    model = LassoCV(cv=len(X_train), random_state=10, max_iter=100000)
    model.fit(X_train, y_train)
    lasso_best = Lasso(alpha=model.alpha_)
    lasso_best.fit(X_train, y_train)
    error = mean_squared_error(y_test, lasso_best.predict(X_test))
    avg_error += error
    coeffs = list(
        lasso_best.coef_,
    )
    all_coeffs.append(coeffs)
    test_score = lasso_best.score(X_test, y_test)
    train_score = lasso_best.score(X_train, y_train)
    model_scores.append(test_score)
    print(test_score)
avg_error /= 10
print("Median Score:", np.median(np.array(sorted(model_scores))))
print(avg_error)

plt.violinplot(model_scores, showmedians=True)
plt.show()

# %%
avg_coeffs = []
for i in range(len(all_coeffs[0])):
    avg = 0
    j = 0
    for coeffs in all_coeffs:
        avg += coeffs[i]
        j += 1
    avg_coeffs.append(avg / j)
print(avg_coeffs)

# %%
avg_error = 0
scores = []
for k in range(200):
    zd_df["Angle of Repose"] = np.random.permutation(zd_df["Angle of Repose"].values)
    y = zd_df["Angle of Repose"]
    for i in range(0, 10):
        X_train = X.drop([(i + j * 10) for j in range(4)])
        y_train = y.drop([(i + j * 10) for j in range(4)])
        X_test = X.drop(X_train.index)
        y_test = y.drop(y_train.index)
        model = LassoCV(cv=len(X_train), random_state=10, max_iter=100000)
        model.fit(X_train, y_train)
        lasso_best = Lasso(alpha=model.alpha_)
        lasso_best.fit(X_train, y_train)
        test_score = lasso_best.score(X_test, y_test)
        scores.append(test_score)
print("Median Score:", np.median(np.array(sorted(scores))))

# %%
plt.ylim(-10, 1)
plt.violinplot([model_scores, scores], showmedians=True, showextrema=True)
plt.axhline(np.percentile(scores, 95), color="r", linestyle="dashed")
print(np.percentile(scores, 95))
plt.show()
