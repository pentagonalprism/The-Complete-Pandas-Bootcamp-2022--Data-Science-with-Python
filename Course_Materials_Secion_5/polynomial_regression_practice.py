#Basic Polynomial Regression

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
#Your dependent variable is changed by the value of your independent variable
lm.fit(X=df.budget.to_frame(),y=df.revenue)


# y = mx+b

#call the slope of the line
slope = lm.coef_

#create an intercept value
intercept = lm.intercept_

#create predictions based on the dependent variable budget
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


def new_dataset():
	budget_new = np.array([[63,66,74,80,85]])
	revenue_new = np.array([74.2,80.7,98.2,94.8,101.7])

	new_movie_data= pd.DataFrame(data = {'budget':budget,'revenue':revenue})

	return new_movie_data

df_new = new_dataset()

#Create a new predicted value for the values in your new dataset based on the old dataset
#past performance can indicate future performance lol
df_new['pred'] = lm.predict(df_new.budget.to_frame())


def scatter_plot_regr(movie_data,x_lin,y_lin):
	plt.figure(figsize=(12,8))
	plt.scatter(x=df.budget, y=df.revenue)
	plt.plot(x_lin,y_lin,c='red',label='regression')
	plt.scatter(x=df_new.budget, y=df_new.revenue)
	plt.xlabel(df.columns.values[1])
	plt.ylabel(df.columns.values[0])
	return plt.show()


poly_m = np.polyfit(x=movie_data.budget, y=movie_data.revenue,deg=9)

x_poly = np.linspace(0,100,1000)
y_poly = np.polyval(poly_m,x_poly)

def scatter_plot_polyregr(movie_data,x_lin,y_lin):
	plt.figure(figsize=(12,8))
	plt.scatter(x=df.budget, y=df.revenue)
	plt.plot(x_lin,y_lin,c='red',label='regression')
	plt.scatter(x=df_new.budget, y=df_new.revenue)
	plt.plot(x_poly,y_poly, label = 'Polynomial Regression | Degree = 9 (Overfit)',linestyle='--',color='Orange')
	plt.xlabel(df.columns.values[1])
	plt.ylabel(df.columns.values[0])
	plt.legend(fontsize = 11, loc = 4)
	plt.ylim(0, 150)
	return plt.show()

scatter_plot_polyregr(movie_data,x_lin,y_lin)