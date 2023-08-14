from sklearn import tree

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# 1 => smooth, 2 => bumpy
labels = [0, 0, 1, 1]
# 0 => apples, 1 = oranges

clf = tree.DecisionTreeClassifier()  # classifier
clf = clf.fit(features, labels)
print(clf.predict([[160, 0]]))
