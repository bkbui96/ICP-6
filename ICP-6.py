import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

#Read data and only get values < 1200
train = pd.read_csv('train.csv')
train = train[train['GarageArea'] < 1200]

#Set variables to plot data
GarageArea = train.GarageArea
SalePrice = train.SalePrice


#Plotting data
plt.scatter(GarageArea, SalePrice, alpha=.75, color='b') #alpha helps to show overlapping data

plt.xlabel('Garage Area')
plt.ylabel('Sale Price')
plt.title('Linear Regression Model')
plt.show()
