
#RF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
df=pd.read_csv('wisconsin-orginal.csv')
features=list(df.columns)
print(features)
df = df.replace("?", np.nan) 
df = df.dropna()
x=features[1:-1]
X=features[1:-1]
y=features[-1]
Y=features[-1]
train, test = train_test_split(df, test_size=0.2)
X = np.array([[train[f].tolist()[i] for f in X] for i in range(len(train))])
Y = np.array([i for i in train[Y]])
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X,Y)
fn=x
cn=y
fig, axes = plt.subplots(nrows = 1,ncols = 10,figsize = (10,2), dpi=900)
for index in range(0,10):
    tree.plot_tree(clf.estimators_[index],
                   feature_names = fn, 
                   class_names=cn,
                   filled = True,
                   ax = axes[index]);

    axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
fig.savefig('rf_5trees.png')