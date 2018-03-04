import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

usedAttributes = ["attack", "base_total", "sp_attack", "sp_defense"]  # Define selected attributes.
pokemonData = pd.read_csv('pokemon.csv', usecols=usedAttributes)  # Read CSV file.

for i in range(4):      # Data Visualization
    plt.show(sns.distplot(pokemonData[usedAttributes[i]])) # Histogram

    plt.xlabel(usedAttributes[i])  # X label for Box plot
    plt.show(plt.boxplot(pokemonData[usedAttributes[i]])) # Box Plot

    if( i != 0):
        plt.xlabel(usedAttributes[i]) # X label for scatter plot
        plt.ylabel(usedAttributes[0]) # Y label for scatter plot
        plt.scatter(pokemonData[usedAttributes[i]], pokemonData[usedAttributes[0]]) # Scatter Plot
        plt.show()

# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn import metrics

lm = LinearRegression()
X = pokemonData[[usedAttributes[1], usedAttributes[2], usedAttributes[3]]]
y = pokemonData[usedAttributes[0]]
sum_error = 0
kf = KFold(n_splits=10)
i = 1
for train, test in kf.split(X):
    #print("%s %s" % (train, test))
    X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    lm.fit(X_train, y_train)
    predictions = lm.predict(X_test)
    error = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    sum_error += error
    print("RMSE-" + str(i) + " = " + str(error))
    i = i+1

coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)
print("Intercept = " + str(lm.intercept_))
print("average RMSE = " + str(sum_error/10))
