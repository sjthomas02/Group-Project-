# Life expectancy project: all code cells from LE_Regression.ipynb, in notebook order
# Markdown notes are not included.


# Standard set up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import os
os.getcwd()


# Importing Life Expectancy dataset
df_le = pd.read_csv("Life_Expectancy.csv")


# Checking duplicates, missing values and summary statistics
all_cols = ["Country", "Region", "Year", "Infant_deaths", "Under_five_deaths",
            "Adult_mortality", "Alcohol_consumption", "Hepatitis_B", "Measles",
            "BMI", "Polio", "Diphtheria", "Incidents_HIV", "GDP_per_capita",
            "Population_mln", "Thinness_ten_nineteen_years", "Thinness_five_nine_years",
            "Schooling", "Economy_status_Developed", "Economy_status_Developing",
            "Life_expectancy"]
print(df_le.shape, df_le["Country"].nunique(), df_le["Year"].nunique())
print(df_le.duplicated(["Country", "Year"]).sum())
print(df_le[all_cols].isna().sum())  
print(df_le[all_cols].describe())


# Looking at identifiers in LE dataset
identify_cols = ["Country", "Region", "Year"]

# Uniqueness
for col in identify_cols:
    print(col, df_le[col].nunique())

# Missingness
print(df_le[identify_cols].count())
print(df_le[identify_cols].isna().sum())

# Count of rows per value (e.g. rows per year, or per region)
for col in identify_cols:
    print(df_le[col].value_counts().sort_index().head(), "\n")


#Data distribution of main variables
main_var = ["Life_expectancy", "GDP_per_capita"]
df_le[main_var].hist(bins=30, figsize=(8, 4))
plt.tight_layout(); plt.show()


# Checking if GDP_per_Capita =0
zero_gdp_per_cap = df_le[df_le["GDP_per_capita"] == 0]
print(len(zero_gdp_per_cap))


# Logging GDP per capita
df_le["ln_gdp"] = np.log(df_le["GDP_per_capita"])

#Data distribution of logged GDP per capita
df_le["ln_gdp"].hist(bins=30, figsize=(8, 4))
plt.tight_layout(); plt.show()


# Visualising Life Expectancy and Log of GDP per capita
variables = {"Life_expectancy": "Life expectancy (years)", "ln_gdp": "ln GDP per capita"}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (var, ylabel) in zip(axes, variables.items()):
    means = df_le.groupby(["Year", "Economy_status_Developed"])[var].mean().unstack()
    means = means.rename(columns={0: "Developing", 1: "Developed"})
    means.plot(ax=ax); ax.set_ylabel(ylabel); ax.set_title(ylabel + " over time")
plt.tight_layout(); plt.show()


# Pooled OLS formula
formula = "Life_expectancy ~ Incidents_HIV + ln_gdp + Schooling + Hepatitis_B"

pooled = smf.ols(formula, data=df_le).fit(
    cov_type="cluster", cov_kwds={"groups": df_le["Country"]})
print(pooled.summary())


# Splitting Pooled OLS by development: Developed countries 
developed = df_le[df_le["Economy_status_Developed"] == 1]

m1 = smf.ols(formula, data=developed).fit(cov_type="cluster", cov_kwds={"groups": developed["Country"]})

print(m1.summary())


# Creating data frame of developing countries
developing = df_le[df_le["Economy_status_Developed"] == 0]


#Analysing spread of HIV incidents in developed countries
print(developed["Incidents_HIV"].describe())
developing["Incidents_HIV"].describe()


# Correlation matrix to investigate if collinearity driving high condition number
developed[["Incidents_HIV", "ln_gdp", "Schooling", "Hepatitis_B"]].corr()


# Standardising variables to test fit (for diagnostic only)
developed["ln_gdp_standard"] = (developed["ln_gdp"] - developed["ln_gdp"].mean()) / developed["ln_gdp"].std()
developed["Incidents_HIV_standard"] = (developed["Incidents_HIV"] - developed["Incidents_HIV"].mean()) / developed["Incidents_HIV"].std()
developed["Schooling_standard"] = (developed["Schooling"] - developed["Schooling"].mean()) / developed["Schooling"].std()
developed["Hepatitis_B_standard"] = (developed["Hepatitis_B"] - developed["Hepatitis_B"].mean()) / developed["Hepatitis_B"].std()

formula_standard = "Life_expectancy ~ Incidents_HIV_standard + ln_gdp_standard + Schooling_standard + Hepatitis_B_standard"

m1_standard = smf.ols(formula_standard, data=developed).fit(cov_type="cluster", cov_kwds={"groups": developed["Country"]})

print(m1_standard.summary())   # developed refit


#Analysing spread of Schooling in developed countries
print(developed["Schooling"].describe())
print(developing["Schooling"].describe())


#Analysing spread of Hepatitis B in developed countries
print(developed["Hepatitis_B"].describe())
print(developing["Hepatitis_B"].describe())


# Correlation matrix analysing link between Schooling, Hep B and life expectancy in developed countries 
developed[["Life_expectancy", "Schooling", "Hepatitis_B"]].corr()


# Correlation matrix analysing link between Schooling, Hep B and life expectancy in developing countries 
developing[["Life_expectancy", "Schooling", "Hepatitis_B"]].corr()


# Splitting Pooled OLS by development: Developing countries
m0 = smf.ols(formula, data=developing).fit(cov_type="cluster", cov_kwds={"groups": developing["Country"]})

print(m0.summary())   # developing


# Comparing Pooled OLS: All data, Developed countries, Developing countries
from statsmodels.iolib.summary2 import summary_col

print(summary_col(
    [pooled, m1, m0],
    model_names=["All", "Developed", "Developing"],
    stars=True,
    info_dict={"N": lambda m: f"{int(m.nobs)}"},
))


# Setting up for lagging variables
df_le = df_le.sort_values(["Country", "Year"])

# Creating lag of ln_gdp (1 year)
df_le["ln_gdp_1"] = df_le.groupby("Country")["ln_gdp"].shift(1)

# Creating lag of ln_gdp (3 years)
df_le["ln_gdp_3"] = df_le.groupby("Country")["ln_gdp"].shift(3)

# Creating lag of ln_gdp (5 years)
df_le["ln_gdp_5"] = df_le.groupby("Country")["ln_gdp"].shift(5)


# Pooled OLS with log of GDP per capita lagged by 1 year
d1 = df_le.dropna(subset=["ln_gdp_1"])

formula_lag1 = "Life_expectancy ~ Incidents_HIV + ln_gdp_1 + Schooling + Hepatitis_B"

pooled_lag1 = smf.ols(formula_lag1, data=d1).fit(
    cov_type="cluster", cov_kwds={"groups": d1["Country"]})
print(pooled_lag1.summary())


# Pooled OLS with log of GDP per capita lagged by 3 years
d3 = df_le.dropna(subset=["ln_gdp_3"])

formula_lag3 = "Life_expectancy ~ Incidents_HIV + ln_gdp_3 + Schooling + Hepatitis_B"

pooled_lag3 = smf.ols(formula_lag3, data=d3).fit(
    cov_type="cluster", cov_kwds={"groups": d3["Country"]})
print(pooled_lag3.summary())


# Pooled OLS with log of GDP per capita lagged by 5 years
d5 = df_le.dropna(subset=["ln_gdp_5"])

formula_lag5 = "Life_expectancy ~ Incidents_HIV + ln_gdp_5 + Schooling + Hepatitis_B"

pooled_lag5 = smf.ols(formula_lag5, data=d5).fit(
    cov_type="cluster", cov_kwds={"groups": d5["Country"]})
print(pooled_lag5.summary())


from statsmodels.iolib.summary2 import summary_col

# Fit the three lag models (drop NaN rows first so the clusters match the rows)
d1 = df_le.dropna(subset=["ln_gdp_1"])
d3 = df_le.dropna(subset=["ln_gdp_3"])
d5 = df_le.dropna(subset=["ln_gdp_5"])

pooled_lag1 = smf.ols("Life_expectancy ~ Incidents_HIV + ln_gdp_1 + Schooling + Hepatitis_B", data=d1).fit(
    cov_type="cluster", cov_kwds={"groups": d1["Country"]})
pooled_lag3 = smf.ols("Life_expectancy ~ Incidents_HIV + ln_gdp_3 + Schooling + Hepatitis_B", data=d3).fit(
    cov_type="cluster", cov_kwds={"groups": d3["Country"]})
pooled_lag5 = smf.ols("Life_expectancy ~ Incidents_HIV + ln_gdp_5 + Schooling + Hepatitis_B", data=d5).fit(
    cov_type="cluster", cov_kwds={"groups": d5["Country"]})

# Side-by-side table
print(summary_col(
    [pooled, pooled_lag1, pooled_lag3, pooled_lag5],
    model_names=["Pooled", "Lag 1", "Lag 3", "Lag 5"],
    stars=True,
    info_dict={"N": lambda m: f"{int(m.nobs)}", "R2": lambda m: f"{m.rsquared:.3f}"},
))


# One common sample: rows where the 5-year lag exists (this also covers lags 1 and 3)
common = df_le.dropna(subset=["ln_gdp_5"])

def fit(formula):
    return smf.ols(formula, data=common).fit(
        cov_type="cluster", cov_kwds={"groups": common["Country"]})

pooled_c = fit("Life_expectancy ~ Incidents_HIV + ln_gdp   + Schooling + Hepatitis_B")
lag1_c   = fit("Life_expectancy ~ Incidents_HIV + ln_gdp_1 + Schooling + Hepatitis_B")
lag3_c   = fit("Life_expectancy ~ Incidents_HIV + ln_gdp_3 + Schooling + Hepatitis_B")
lag5_c   = fit("Life_expectancy ~ Incidents_HIV + ln_gdp_5 + Schooling + Hepatitis_B")

print(summary_col(
    [pooled_c, lag1_c, lag3_c, lag5_c],
    model_names=["Pooled", "Lag 1", "Lag 3", "Lag 5"],
    stars=True,
    info_dict={"N": lambda m: f"{int(m.nobs)}"},
))


# Adding squared term for log of GDP per capita
df_le["ln_gdp_sq"] = df_le["ln_gdp"] ** 2


#Pooled OLS with square of ln gdp included
formula_sq = "Life_expectancy ~ Incidents_HIV + ln_gdp + ln_gdp_sq + Schooling + Hepatitis_B"

pooled_sq = smf.ols(formula_sq, data=df_le).fit(
    cov_type="cluster", cov_kwds={"groups": df_le["Country"]})
print(pooled_sq.summary())


# Correlation matrix to investigate if collinearity driving high condition number
df_le[["Incidents_HIV", "ln_gdp", "ln_gdp_sq", "Schooling", "Hepatitis_B"]].corr()


# Centering ln_gdp before squaring so square isn't as highly correlated
df_le["ln_gdp_c"] = df_le["ln_gdp"] - df_le["ln_gdp"].mean()
df_le["ln_gdp_c_sq"] = df_le["ln_gdp_c"] ** 2


# Checking correlation matrix 
df_le[["Incidents_HIV", "ln_gdp_c", "ln_gdp_c_sq", "Schooling", "Hepatitis_B"]].corr()


# Restimating pooled OLS with centred ln_gdp
formula_sq_c = "Life_expectancy ~ Incidents_HIV + ln_gdp_c + ln_gdp_c_sq + Schooling + Hepatitis_B"

pooled_sq_c = smf.ols(formula_sq_c, data=df_le).fit(
    cov_type="cluster", cov_kwds={"groups": df_le["Country"]})

print(pooled_sq_c.summary())


# Looking at marginal effects 
# Coefficients of ln_gdp_c and ln_gdp_c_sq
b1 = pooled_sq_c.params["ln_gdp_c"]
b2 = pooled_sq_c.params["ln_gdp_c_sq"]

# Looking at 3 points (25, 50 and 75 percentiles)
x25 = df_le["ln_gdp_c"].quantile(0.25)
x_mean = df_le["ln_gdp_c"].mean()
x75 = df_le["ln_gdp_c"].quantile(0.75)

# Slope at each point
print(b1 + 2 * b2 * x25)
print(b1 + 2 * b2 * x_mean)
print(b1 + 2 * b2 * x75)


# Plotting fitted line of Life Expectancy against ln_gdp

# Grid of GDP values from the lowest to the highest
grid = pd.DataFrame({"ln_gdp": np.linspace(df_le["ln_gdp"].min(), df_le["ln_gdp"].max(), 100)})

# Variables needed for the curve plot
grid["ln_gdp_c"] = grid["ln_gdp"] - df_le["ln_gdp"].mean()
grid["ln_gdp_c_sq"] = grid["ln_gdp_c"] ** 2

# Other variables needed from model, setting to the mean to analyse effect of log of GDP when these variables held constant
grid["Incidents_HIV"] = df_le["Incidents_HIV"].mean()
grid["Schooling"] = df_le["Schooling"].mean()
grid["Hepatitis_B"] = df_le["Hepatitis_B"].mean()

# Predicted life expectancy along the curve
grid["fitted"] = pooled_sq_c.predict(grid)

# Plot
plt.scatter(df_le["ln_gdp"], df_le["Life_expectancy"], s=8, alpha=0.3, label="Data")
plt.plot(grid["ln_gdp"], grid["fitted"], color="red", label="Fitted curve")
plt.xlabel("Log of GDP per capita")
plt.ylabel("Life expectancy (years)")
plt.legend()
plt.show()


# Calculating the turning point
x_turn = -b1 / (2 * b2)
print(x_turn)


# Converting turning point back to ln_gdp (non-centred variable)
ln_gdp_turn = x_turn + df_le["ln_gdp"].mean()
print(ln_gdp_turn)


# Converting turning point of ln_gdp back to before the log
print(np.exp(ln_gdp_turn))


# Looking at maximum values in data
print(df_le["ln_gdp"].max())
print(df_le["GDP_per_capita"].max())


print(formula)


# Country-level fixed effects
fe_country = smf.ols(formula_sq_c + " + C(Country)", data=df_le).fit(
    cov_type="cluster", cov_kwds={"groups": df_le["Country"]})


# Year-level fixed effects
fe_year = smf.ols(formula_sq_c + " + C(Year)", data=df_le).fit(
    cov_type="cluster", cov_kwds={"groups": df_le["Country"]})


# Two-way fixed effects (country and year)
fe_two = smf.ols(formula_sq_c + " + C(Country) + C(Year)", data=df_le).fit(
    cov_type="cluster", cov_kwds={"groups": df_le["Country"]})


# Comparison table with pooled OLS, showing only the main variables
print(summary_col(
    [pooled_sq_c, fe_country, fe_year, fe_two],
    model_names=["Pooled", "Country FE", "Year FE", "Two-way FE"],
    stars=True,
    regressor_order=["Incidents_HIV", "ln_gdp_c", "ln_gdp_c_sq", "Schooling", "Hepatitis_B"],
    drop_omitted=True,
    info_dict={"N": lambda m: f"{int(m.nobs)}"},
))


# Analysing slope of life expectancy on ln_gdp at different GDP per capita levels
b1_fe = fe_two.params["ln_gdp_c"]
b2_fe = fe_two.params["ln_gdp_c_sq"]

print(b1_fe + 2 * b2_fe * x25)
print(b1_fe + 2 * b2_fe * x_mean)
print(b1_fe + 2 * b2_fe * x75)


# Turning point for two-way fixed effects model
x_turn_fe = -b1_fe / (2 * b2_fe)
print(x_turn_fe)

# Converting back to ln_gdp (non-centred variable)
ln_gdp_turn_fe = x_turn_fe + df_le["ln_gdp"].mean()
print(ln_gdp_turn_fe)

# Converting back to GDP per capita (before the log)
print(np.exp(ln_gdp_turn_fe))
