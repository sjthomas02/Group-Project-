[CODE]
# Standard set up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import os
os.getcwd()

[CODE]
# Importing Life Expectancy dataset
df_le = pd.read_csv("Life_Expectancy.csv")

[CODE]
print(df_le.shape, df_le["Country"].nunique(), df_le["Year"].nunique())
print(df_le.duplicated(["Country", "Year"]).sum())      # duplicate country-years
print(df_le[cols].isna().sum())                          # missing values
print(df_le[cols].describe()) 

[CODE]
#Data distribution of main variables
cols = ["Life_expectancy", "GDP_per_capita"]
df_le[cols].hist(bins=30, figsize=(8, 4))
plt.tight_layout(); plt.show()

[MARKDOWN]
**Note** Left skew for life expectancy; right skew for GDP per capita. Need to check if gdp_per_capita has any = 0 which matters if decide to log GDP per capita

[CODE]
# Checking if GDP_per_Capita =0
zero_gdp_per_cap = df_le[df_le["GDP_per_capita"] == 0]
print(len(zero_gdp_per_cap))

[MARKDOWN]
**Note** No values in GDP per capita are equal to zero

[CODE]
# Logging GDP per capita
df_le["ln_gdp"] = np.log(df_le["GDP_per_capita"])

#Data distribution of logged GDP per capita
df_le["ln_gdp"].hist(bins=30, figsize=(8, 4))
plt.tight_layout(); plt.show()

[MARKDOWN]
**Note** Log of GDP per capita looks more normally distributed for analysis.

[MARKDOWN]
**Note** Model below is Pooled OLS. GDP is in per capita terms so no need to include control for population, infant deaths and adult mortality not included as these as proxys for life expectancy

[MARKDOWN]
**Note** Clustering of SE used as pooled OLS means each country appears 16 times and the effects of each country year on year are most likely correlated. Therefore, clustering groups observations by country, but assumes the clusters are independent of eachother.

[CODE]
formula = "Life_expectancy ~ Incidents_HIV + ln_gdp + Schooling + Hepatitis_B"

pooled = smf.ols(formula, data=df_le).fit(
    cov_type="cluster", cov_kwds={"groups": df_le["Country"]})
print(pooled.summary())

[MARKDOWN]
**Note** Below explores splitting the model by developed & undeveloped countries

[CODE]
# Splitting Pooled OLS by development
developed = df_le[df_le["Economy_status_Developed"] == 1]

m1 = smf.ols(formula, data=developed).fit(cov_type="cluster", cov_kwds={"groups": developed["Country"]})

print(m1.summary())   # developed

[CODE]
# Splitting Pooled OLS by development
developing = df_le[df_le["Economy_status_Developed"] == 0]

m0 = smf.ols(formula, data=developing).fit(cov_type="cluster", cov_kwds={"groups": developing["Country"]})

print(m0.summary())   # developing
