#Logistic regression
import pandas as pd
import numpy as np
from itertools import combinations

THRESHOLD = 0.7

def checkCollinear(X, Y):
    corr = np.cov(X, Y)[0][1]/(np.std(X)*np.std(Y))
    if(corr > THRESHOLD or corr < -THRESHOLD):
        return True
    else:
        return False

df=pd.read_csv('wisconsin.csv')

df.replace(to_replace = 'M', value = 0,inplace = True)
df.replace(to_replace = 'B', value = 1,inplace = True)

print(df)

features=list(df.columns)
features=features[2:32]
#print(features)

results = [list(x) for x in combinations(features, 2)]
for i in results:
     x=df[i[0]].tolist()
     y=df[i[1]].tolist()
     flag = checkCollinear(x,y)
     if(flag==True):
           if(i[0] in features):
               features.remove(i[0])
     #rho=np.corrcoef(x,y)
     #if(not(rho[1][0]<0.7 and rho[1][0]>-0.7)):
         #if(i[0] in features):
             #features.remove(i[0])      
      
print(features)
print(len(features))
N = len(df)
X,Y = [],[]
for i in range(N):
    lis = [1]
    for j in features:
        lis.append(df[j][i])
    X.append(lis)
    Y.append(df['diagnosis'][i])

X=np.array(X)
Y=np.array([[i] for i in Y])

Wold = [0 for i in range(len(features)+1)]
Wnew =[0 for i in range(len(features)+1)]

eta = 0.001
yHat = []
Wold=np.array(Wold);
Wnew=np.array(Wnew);
i=0
while(True):
    for i in range(0,len(X)):
        temp=np.dot(np.transpose(Wold),X[i])  
        sig = 1/(1+np.exp(-temp))
        if(sig > 0.5):
            yHat.append(1)
        else:
            yHat.append(0)
        if(Y[i] != yHat[i]):
            Wnew = Wold + (eta*sig*(Y[i] - sig)*(1 - sig))*X[i]
            Wold = Wnew  
    if(np.array_equal(Wold, Wnew)):
        break
    i+=1    
        
print(Wnew)
#print(i)

'''
#Logistic reg km
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"wisconsin.csv")
# print(df)

del df['id']

df.head()

features = df.columns
features = list(features)
features = features[1:len(features)-1]
print(len(features))

# removing multicollinearity
final_features = [x for x in features]
CR = df[features].corr().values.tolist()

for i in range(len(CR)):
    for j in range(i+1,len(CR)):
        if abs(CR[i][j])>0.7 and features[i] in final_features:
            final_features.remove(features[i])

# final_features.append('diagnosis')
print(final_features)

features_list = []
for i in range(0,len(final_features)):
    features_list.append(np.array(df[final_features[i]]))
    
y = df['diagnosis']

for i in range(len(y)):
    if y[i]=='B':
        y[i]=0
    elif y[i]=='M':
        y[i]=1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_regression(x,y):
    W = np.random.rand(x.shape[1])
    convergence = False
    eta = 0.0001
    DATAPOINTS_SIZE = len(x)
    
    y_cap = []
    count = 0
    while (count<10000):
        wTx = W[0]*1
#         congergence = True
        sigmoid_values=[]
        for i in range(0,DATAPOINTS_SIZE):
            wTx = W.dot(x[i])
            sig_value = sigmoid(wTx)
            if(sig_value>=eta):
                y_cap.append(1)
            else:
                y_cap.append(0)
            
            if y_cap[i] != y[i]:
                W = W + (y[i]-sig_value)*eta*x[i]
                convergence = False
        count+=1
    print(W)

x_data = df[final_features[0:len(final_features)-1]]
y_data = y

X_train,X_test,y_train,y_test = train_test_split(x_data.to_numpy(), y_data.to_numpy(), test_size=0.33)
  
print(X_train,y_train)                                                 
logistic_regression(X_train,y_train)     
'''                        

'''
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


THRESHOLD = 0.7

# Outlier Treatment
def outlier_treatment(df, feature):
    q1, q3 = np.percentile(df[feature], [25, 75])
    IQR = q3 - q1 
    lower_range = q1 - (3 * IQR) 
    upper_range = q3 + (3 * IQR)
    to_drop = df[(df[feature]<lower_range)|(df[feature]>upper_range)]
    df.drop(to_drop.index, inplace=True)

def calc(X, Y, W, theta):
    TP, TN, FP, FN = 0, 0, 0, 0
    Yhat = np.dot(X, W)
    N = len(X)
    for i in range(N):
        Ycap = 1 / (1 + math.exp(-Yhat[i]))

        if(Ycap >= theta):
            Ycap = 1
        else:
            Ycap = 0

        if(Ycap == 1 and Y[i] == 1):
            TP += 1
        elif(Ycap == 0 and Y[i] == 1):
            FN += 1
        elif(Ycap == 1 and Y[i] == 0):
            FP += 1
        else:
            TN += 1
           
    return [TP, FN, FP, TN]



df = pd.read_csv('data.csv')

# create a scaler object
std_scaler = StandardScaler()
std_scaler
# fit and transform the data
df_std = pd.DataFrame(std_scaler.fit_transform(df), columns=df.columns)

cols = list(df.columns)[2:32]

features = list(cols)

corr = df[cols].corr().values

for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        if(abs(corr[i][j]) >= THRESHOLD and cols[i] in features and cols[j] in features):
            features.remove(cols[i])

# print(len(cols), len(features))
print(features)

train, test = train_test_split(df, test_size = 0.2)

trainSize = len(train)
train = train.reindex([i for i in range(0, trainSize)])

testSize = len(test)
test = test.reindex([i for i in range(0, testSize)])

print(trainSize, testSize)

# 1 - Malignant, 0 - Benign

X, Y = [], []

for i in range(trainSize):
    lis = [1]
    for j in features:
        lis.append(train[j][i])
    X.append(lis)
    if train['diagnosis'][i] == "M":
        Y.append(1)
    else:
        Y.append(0)

featureSize = len(features)
     
Wold, Wnew = [0 for i in range(featureSize+1)], [0 for i in range(featureSize+1)]
Wold, Wnew = np.array(Wold), np.array(Wnew)
X, Y = np.array(X), np.array(Y)
gradient = [0 for _ in range(len(X))]
eeta = 0.007

# iterations = 1000000

# while(True):
#     iterations-=1
#     for i in range(trainSize):
#         Yhat = X.dot(Wold)
#         sigmoid = 1 / (1 + math.exp(-1*Yhat[i]))
#         if(sigmoid >= 0.5):
#             Ycap = 1
#         else:
#             Ycap = 0
#         if(Ycap != Y[i]):
#             gradient = (Y[i] - sigmoid) * sigmoid * (1 - sigmoid) * X[i]
#             Wnew = Wold + eeta*gradient
#             Wold = Wnew

#     print(Wold)
#     if (np.all(gradient) == 0 or iterations == 0):
#             break
 
Wold = [ -1.56321027, -7.41984046, 0.19201655, 0.80786062, 4.56161812, -3.86185968,
         -1.37173668, 0.03099825, 0.20293206, 16.14072523, 2.4543871, -12.90958941 ]

# Wold = [-14.13967032, 7.95389434, -1.09796794, 5.13651019, -1.44968948, 14.10065082,
#         -2.88448603, 0.28892423, -5.05859146, 79.52821553, 0.547714, -46.33737769]

print("Coefficients")      
print(Wold)


XTest, YTest = [], []
for i in range(testSize):
    lis = [1]
    for j in features:
        lis.append(test[j][i])
    XTest.append(lis)
    if test['diagnosis'][i] == "M":
        YTest.append(1)
    else:
        YTest.append(0)

TP, FN, FP, TN = calc(XTest, YTest, Wold, 0.5)

print(TP, FN, FP, TN)

P = TP/(TP+FP)
R = TP/(TP+FN)

print("Accuracy: ", (TP+TN)/testSize)

print("Precision: ", P)

print("Recall: ", R)

print("F measure: ", 2*P*R/(P+R))

print("TPR: ", TP/(TP + FN))

print("FPR: ", FP/(FP + TN))


theta = np.linspace(0, 1, 10000)

TPR, FPR = [], []

for i in theta:
    TP, FN, FP, TN = calc(XTest, YTest, Wold, i)
    TPR.append(TP/(TP + FN))
    FPR.append(FP/(FP + TN))

FPR.extend([0, 1])
TPR.extend([0, 1])

FPR, TPR = zip(*sorted(zip(FPR, TPR)))

AUC = 0

for i in range(len(FPR)-1):
    AUC += 0.5*(FPR[i+1]-FPR[i])*(TPR[i]+TPR[i+1])
   
print("Area under the curve: ", AUC)

# print("Area under the curve: ", np.trapz(TPR, FPR))

plt.plot(FPR, TPR)
plt.show()
'''                    