import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as pyplot

# read data
dataframe = pd.read_fwf('brain_body.txt') 
# read a data set that contains the average brain
# and body weight for a number of animal species

x_values = dataframe[['brain']]
y_values = dataframe[['body']]
# our goal is that given a new animal's body weight 
# we'll be able to predict what its brain size is

# train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

# visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()