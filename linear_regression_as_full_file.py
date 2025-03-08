# %%
# %%

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import sklearn.linear_model as linear
import numpy as np

# %%
# %%

df = pd.read_csv("compiled_data.csv")
df

# %%
# %%

from scipy import stats

ys = list(df["Angle of Repose"])
print(ys)

# %%
# %%

z_scored_ys = stats.zscore(ys)

# %%
# %%

xs = []
for i in range(1, len(list(df.iloc[1:,]))):
    xs.append([list(df.iloc[i - 1 :, i])[j] for j in range(0, len(ys), 10)])
# for i in range(len(xs)):
#     xs[i]=[val**0.25 for val in xs[i]]
zd_xs = [list(stats.zscore(x)) for x in xs]

# %%
# %%

new_skeleton = [[] for i in range(len(xs))]
for i, column in enumerate(zd_xs):
    print(column)
    for cell in column:
        for j in range(10):
            new_skeleton[i].append(cell)

# %%
# %%

zd_df = pd.DataFrame(
    {
        "Angle of Repose": z_scored_ys,
        "Length to Width Ratio": new_skeleton[0],
        "Coeff of Friction": new_skeleton[1],
        "Mass": new_skeleton[2],
        "Volume": new_skeleton[3],
        "Length to Width Ratio s.d.": new_skeleton[4],
        "Coeff of Friction s.d.": new_skeleton[5],
        "Volume s.d.": new_skeleton[6],
        "Mass s.d.": new_skeleton[7],
    }
)
zd_df

# %%
# %%
plt.rcParams.update({"font.size": 14})
fig, axs = plt.subplots(2, 2, figsize=(7, 5))
x_ = ["Mass", "Mass s.d.", "Volume", "Volume s.d."]
total = 0
for j in range(2):
    total += j
    for k in range(2):
        total += k
        print(total)
        x = x_[total]
        axs[j, k].scatter(df[x], df["Angle of Repose"], c="k", alpha=0.1)
        average_y_groups = [
            np.mean(np.array(df["Angle of Repose"][10 * i : 10 * i + 10 - 1]))
            for i in range(len(df["Angle of Repose"]) // 10)
        ]
        axs[j, k].scatter(
            [df[x][i] for i in range(0, 50, 10)], average_y_groups, c="r", s=100
        )
        axs[j, k].set_xlabel(["Mass", "Mass s.d.", "Volume", "Volume s.d."][total])
        axs[j, k].set_ylabel("AOR (Degrees)")
fig.tight_layout(pad=1.15)
plt.show()

# %%


# %%

y = zd_df["Angle of Repose"]
X = zd_df.drop(["Angle of Repose"], axis=1).astype("float64")


# %%
model = linear.LassoCV(cv=10, random_state=0, max_iter=100000)

# Fit model
model.fit(X, y)
plt.semilogx(model.alphas_, model.mse_path_, ":")
plt.plot(
    model.alphas_,
    model.mse_path_.mean(axis=-1),
    "k",
    linewidth=2,
)
plt.axvline(
    model.alpha_,
    linestyle="--",
    color="k",
)


plt.xlabel("Alpha")
plt.ylabel("Mean square error")
plt.title("Mean square error on each fold")
plt.axis("tight")

ymin, ymax = 0, 1.75
plt.ylim(ymin, ymax)

# %%
r_squareds = []
coeffs_all_models = []
for i in range(8):
    lasso_best = linear.Lasso(alpha=0.032600090499721894)
    lasso_best.fit(X, y)
    print(lasso_best.get_params())
    coeffs = list(
        lasso_best.coef_,
    )
    train_score = lasso_best.score(X, y)
    r_squareds.append(train_score)
    coeffs_all_models.append(
        list(zip([column.replace(" ", "\n") for column in X.columns], coeffs))
    )
    print("Model", i + 1)
    print("R Squared:", train_score)
    print(list(zip([column.replace(" ", "\n") for column in X.columns], coeffs)))
    coeffs = [abs(coeff) for coeff in coeffs]
    print("Dropping", X.columns[coeffs.index(max(coeffs))])
    X = X.drop(X.columns[coeffs.index(max(coeffs))], axis=1)

    print("\n")


# %%
plt.rcParams.update({"font.size": 14})
x = [i for i in range(8)]
plt.plot(x, r_squareds, linewidth=3)
plt.scatter(x, r_squareds, c="r")
plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
plt.xlabel("Number of Features Removed from Model")
plt.ylabel("R-Squared Value of Model")
plt.title("Model-explained variance with feature elimination")
plt.show()

# %%
plt.rcParams.update({"font.size": 14})
fig, ax = plt.subplots(figsize=(7, 4))
ax.set_ylim(-0.6, 0.6)
local_x = [coeff[0] for coeff in coeffs_all_models[0]]
local_y = [coeff[1] for coeff in coeffs_all_models[0]]
ax.axhline(0, linestyle="dotted", c="k")
ax.scatter(local_x, local_y, c="r", marker="+", s=200)
ax.set_xlabel("Attributes")
ax.set_ylabel("Coefficient Values")
ax.set_title("Coefficients for multiple linear regression model")
plt.show()

# %%
wet_AORS = [
    36.70639496,
    44.14815339,
    42.31637803,
    41.7178955,
    49.13631135,
    40.71381502,
    41.92704327,
    47.57090818,
    32.42581098,
    44.22456394,
]

fig, ax = plt.subplots(figsize=(3, 5))
dry_AORs = list(df["Angle of Repose"][40:])
xs = ["Pile" for i in range(10)] + ["Submerged Pile" for i in range(10)]
ax.scatter(xs, dry_AORs + wet_AORS, c="k", alpha=0.1)
ax.scatter([""], 45, alpha=0)
ax.scatter(
    ["Pile", "Submerged Pile"],
    [np.mean(np.array(dry_AORs)), np.mean(np.array(wet_AORS))],
    c="r",
    s=200,
)
ax.set_ylabel("AOR")
ax.set_xlabel("Pile Types")
print(dry_AORs)
plt.show()

# %%
