import pandas as pd
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

#Read in data
train = pd.read_csv('weatherHistory.csv')

#Drop nulls
train = train.dropna()

#Convert Categorical to Null
from sklearn.preprocessing import LabelEncoder
train = train.apply(LabelEncoder().fit_transform)



##Build a linear model
y = train['Temperature (C)']
X = train.drop(['Temperature (C)'], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
##Evaluate the performance and visualize results
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))