import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn import svm
from matplotlib import style

style.use("ggplot")

x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

if len(x) != len(y):
    print("No. of elements in x is not equal to no. of elements"
          " in y. Exiting ...")
    sys.exit(1)

plt.scatter(x, y)
# plt.show()

coordinates = []
for i in range(len(x)):
    coordinates.append([x[i], y[i]])

# print(coordinates)

X = np.array(coordinates)

Y = [0, 1, 0, 1, 0, 1]

classifier = svm.SVC(kernel='linear', C=1.0)
classifier.fit(X, Y)

# print(classifier.predict([[0.58, 0.76]]))
# print(classifier.predict([[10.58, 10.76]]))

w = classifier.coef_[0]
print(w)

a = -w[0]/w[1]

xx = np.linspace(0, 12)
yy = a * xx - classifier.intercept_[0] / w[1]

# k = black, - = line
h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.legend()
plt.show()
