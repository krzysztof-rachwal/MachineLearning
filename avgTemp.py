import matplotlib.pyplot as plt #for visual representation
import seaborn as sns
import pandas as pd

nyc = pd.read_csv('avgTemp1895-2018.csv') #load data file and create dataframe 

print(nyc.head(3))

#Formatting data we do not need month in date column
nyc.columns = ['Date', 'Temperature', 'Anomaly']
nyc.Date = nyc.Date.floordiv(100)#divide all numeric values in Date column by 100 which chops off last 2 digits (month)

print(nyc.head(3))#formatted data

#splitting data for training and testing
from sklearn.model_selection import train_test_split
#first argument: smaple data to train with 
#second argument: target set of values in this case temperatures in data frame
#random_state arg repetability it can be any value. But alvays use the same values to get the same resuts
X_train, X_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1, 1), nyc.Temperature.values, #(-1, 1) specifies no of columns and rows -1 gives give no of rows there should be based on the no of columns
    random_state=11)

print(X_train.shape)
print(X_test.shape)

#Training the model 
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

print(linear_regression.fit(X=X_train, y=y_train)) #actual training
print(linear_regression.coef_)#array of coefficients for each training dataset
print(linear_regression.intercept_)
#we can use above values to make predictions for future years.

#testing the model 
predicted = linear_regression.predict(X_test)

expected = y_test

#for loop to zip together every fifth predictted and expected value
for p, e in zip(predicted[::5], expected[::5]):
    print(f'Predicted: {p:2f}, Expected: {e:2f}')

#predicting future temperatures and estimating past temperatures

predict = (lambda x: linear_regression.coef_ * x +
            linear_regression.intercept_)

print(predict(2019))
print(predict(1890))

#visualising the dataset with the regression line
#data - our dataframe (nyc), what columns to use on x and y axis, hue - use temp values to colour the graph, palette - specify the olour, no legend
axes = sns.scatterplot(data=nyc, x='Date', y='Temperature', hue='Temperature', palette='winter', legend=False)
#print(plt.show())

#change scale on y axis
axes.set_ylim(10,70)
#print(plt.show())

#draw a regression line
import numpy as np

#create an array from a min and max values in a Dates list
x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
y = predict(x)
print(y)

line = plt.plot(x, y)

print(plt.show())