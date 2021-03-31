import numpy as np
import pandas as pd
from sklearn import neighbors,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

data=pd.read_csv('car data')
X=data[[
    "buying",
    "maint",
    "safety"
]].values
y=data[["class"]]
Le=LabelEncoder()
for i in range(len(X[0])):
    X[:,i]=Le.fit_transform(X[:,i])
label_mapping={
    'unacc':3,
    'acc':2,
    'good':1,
    'vgood':0
}
y=y['class'].map(label_mapping)
y=np.array(y)
knn=neighbors.KNeighborsClassifier(n_neighbors=25,weights='uniform')
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42 )
knn.fit(X_train,y_train)
prediction=knn.predict(X_test)
accuracy=metrics.accuracy_score(y_test,prediction)
print("predictions:",prediction)
print("accuracy:",accuracy)
a=1727
print("actual value:",y[a])
print("predicted value:",knn.predict(X)[a])
pickle.dump(knn,open('model.pickle','wb'))
model=pickle.load,open('model.pickle','rb')