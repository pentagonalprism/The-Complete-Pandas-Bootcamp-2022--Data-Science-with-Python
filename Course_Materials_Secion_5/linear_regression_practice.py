#Basic Linear Regression

import pandas as pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


budget = np.array([5,10,17,27,35,40,42,49,54,60])

revenue = np.array([2.6,19.0,23.8,26.9,41.1,58.3,40.3,58.7,73.1,69.7])

movie_data = pd.DataFrame(data={'budget':budget, 'revenue':revenue})

#Create a Basic Scatter Plot
def scatter_plot(movie_data):
	plt.figure(figsize=(12,8))
	plt.scatter(x=df.budget, y=df.revenue)
	plt.xlabel(df.columns.values[1])
	plt.ylabel(df.columns.values[0])
	return plt.show()

#Create a Linear Regression
lm = LinearRegression(fit_intercept = True)

#Fit the Budget to the be your dependent variable (x) and revenue to independent (y) 
lm.fit(X=df.budget.to_frame(),y=df.revenue)


# y = mx+b

#call the slope of the line
slope = lm.coef_

#create an intercept value
intercept = lm.intercept_

df['pred'] = lm.predict(df.budget.to_frame())

x_lin = np.array([0,100])
y_lin = intercept + slope + x_lin	

def scatter_plot_regr(movie_data,x_lin,y_lin):
	plt.figure(figsize=(12,8))
	plt.scatter(x=df.budget, y=df.revenue)
	plt.plot(x_lin,y_lin,c='red',label='regression')
	plt.xlabel(df.columns.values[1])
	plt.ylabel(df.columns.values[0])
	return plt.show()

scatter_plot_regr(movie_data,x_lin,y_lin)