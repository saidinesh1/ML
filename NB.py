#Naive Bayes
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def findPriors(train, output):
    return train.groupby(output).size().div(len(train))


def constructCPT(train, features, output):
    class_probs = findPriors(train, output)
    cpt = dict()
    for i in features:
        cpt[i] = train.groupby([i, output]).size().div(len(train)).div(class_probs)
    return cpt
    
    
def predict(test, cpt, label):
    Ypred = []
    class_probs = findPriors(train, output)
    
    for i in range(len(test)):
        X = test[features].iloc[i].tolist()
        probs = []
        for j in label:
            prob = 1
            for i in range(len(X)):
                _cpt = cpt[features[i]]
                if (X[i], j) in _cpt.index:
                    prob *= _cpt[X[i]][j]
            prob *= class_probs[j]
            probs.append((prob, j))
        
        Ypred.append(max(probs)[1])
    return Ypred 


def calcMisclass(Y, Ypred):
    res = 0
    for i in range(len(Y)):
        if Y[i]!=Ypred[i]:
            res+=1
    return res



df = pd.read_csv(r'wisconsin-orginal.csv', sep=",")

N = len(df)

# (2 for benign, 4 for malignant)

features = [df.columns[i] for i in range(1, 10)]
output = 'Class'

print("FEATURES: ", features)

#  replacing ? with mean

for i in features:
    data = [x for x in df[i] if x!='?']
    data = list(map(int, data))
    mean =  round(np.mean(data))
    df[i] = df[i].replace('?', mean)

df = df.apply(pd.to_numeric)

train, test = train_test_split(df, test_size = 0.2)

trainSize = len(test)
testSize = len(train)
    
c1 = df['Class'].value_counts()[2]
c2 = df['Class'].value_counts()[4]

print(c1, c2)

cpt = constructCPT(train, features, output)

label = df[output].unique()

Y = test[output].tolist()
Ypred = predict(test, cpt, label)

print("[Size of testing set] ", len(test))
print("[Misclassifications] ", calcMisclass(Y, Ypred))

'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"wisconsin-orginal.csv")
#del df['id']

features = df.columns
features = list(features)
features = features[0:len(features)-1]


def remove_multicollinearity(features):    
    final_features = [x for x in features]
    CR = df[features].corr().values.tolist()

    for i in range(len(CR)):
        for j in range(i+1,len(CR)):
            if abs(CR[i][j])>0.7 and features[i] in final_features:
                final_features.remove(features[i])

    return final_features

final_features = remove_multicollinearity(features)
print(final_features)

features_list = []
for i in range(0,len(final_features)):
    features_list.append(np.array(df[final_features[i]]))

y = df['Class']
for i in range(len(y)):
    if y[i]==2:
        y[i]=0
    elif y[i]==4:
        y[i]=1

final_features.remove('bare_nucleoli') #dataset char problem
x_data = df[final_features]
y_data = pd.DataFrame(y)

def makeCPT(x,y,unique_quantities):
    y = list(y)
    
    CPT_YES_FEATURES={}
    CPT_NO_FEATURES = {}
    for k in range(len(x[0])):
        CPT_YES = {}
        CPT_NO = {}
        for j in unique_quantities[k]:
            yeslist = []
            nolist = []
            for i in range(len(x)):
                if(x[i][k]==j and y[i]==1):
                    yeslist.append(1)
                if(x[i][k]==j and y[i]==0):
                    nolist.append(0)
            CPT_YES[j] = len(yeslist)/y.count(1)
            CPT_NO[j] = len(nolist)/y.count(0)
        CPT_YES_FEATURES[final_features[k]]=CPT_YES
        CPT_NO_FEATURES[final_features[k]]=CPT_NO
    
    return (CPT_YES_FEATURES,CPT_NO_FEATURES)

def testNaiveBayes(x_test,y_test,yCPT,nCPT,y_prob):
    print(x_test)
    Y_posterior = []
    for i in range(len(x_test)):
        num = y_prob
        denom = (1-y_prob)
        for j in range(len(x_test[0])):
            num *= yCPT[final_features[j]][x_test[i][j]]
            denom *= nCPT[final_features[j]][x_test[i][j]]
        denom+=num
        posterior_prob = num/denom
#             print(yCPT[final_features[j]][x_test[i][j]]) 
        Y_posterior.append(posterior_prob)
        
    print(Y_posterior)
    
    for i in range(len(Y_posterior)):
        if (Y_posterior[i]<0.5):
            Y_posterior[i]=0
        else:
            Y_posterior[i]=1
    
    match_count = 0
    for i in range(len(y_test)):
        if (Y_posterior[i]==y_test[i]):
            match_count +=1
    
    acc = match_count/len(y_test)
    
    return acc

def naiveBayesClassifier(x_train,y_train,x_test,y_test):
    no_of_features = len(x_train[0])
#     identifying unique quantities
    unique_quantities = [set()]*6
    
    for j in range(no_of_features):
        for i in range(len(x_train)):
            unique_quantities[j].add(x_train[i][j])

    CPTtup = makeCPT(x_train,y_train,unique_quantities)
    
    yCPT = CPTtup[0]
    nCPT = CPTtup[1]
    print(yCPT)
    print(nCPT)
    y_prob = list(y_train).count(1)/len(y_train)
    print(y_prob)
    accuracy = testNaiveBayes(x_test,y_test,yCPT,nCPT,y_prob)
    
    return accuracy
    pass

X_train,X_test,Y_train,Y_test = train_test_split(x_data.to_numpy(),y_data.to_numpy(), test_size=0.33)

print(np.count_nonzero(Y_train == 1),len(Y_train))
# print(X_train,Y_train)

accuracy = naiveBayesClassifier(X_train,Y_train,X_test,Y_test)

# accuracy of the model
print(accuracy*100)
'''