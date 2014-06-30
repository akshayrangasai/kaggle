from sklearn.ensemble import RandomForestClassifier
import csv as c
import numpy as np

test = np.genfromtxt('train.csv', delimiter=',')
label = test[:,0]
data = test[:,1::]
classifier = RandomForestClassifier(n_estimators = 28)
classifier.fit(data, label)
train = np.genfromtxt('test.csv',delimiter=',')
#print n.shape(data), np.shape(label)
ret = classifier.predict(train)
with open('soln.csv','w') as sol:
    sol.write('ImageId,Label\n')
    i = 1
    for k in ret:
        sol.write('%d,%d\n' % (i , k))
        i = i+1
np.savetxt('soln.txt',ret.astype(int))
