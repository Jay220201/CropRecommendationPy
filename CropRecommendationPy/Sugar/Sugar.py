# Importing essential libraries
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings(action='ignore',
                        category=DeprecationWarning,
                        module='Numpy')
# Loading the dataset
df = pd.read_csv('./Sugar.csv')



df.head()

print("dimension of diabetes data: {}".format(df.shape))




import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(df['PLACE'],label="Count")
plt.show()




df.PLACE=df.PLACE.map({'ARIYALUR':0,
                       'COIMBATORE':1,
                       'CUDDALORE':2,
                       'DHARMAPURI':3,
                       'DINDIGUL':4,
                       'ERODE':5,
                       'KANCHIPURAM':6,
                       'KARUR':7,
                       'KRISHNAGIRI':8,
                       'MADURAI':9,
                       'NAGAPATTINAM':10,
                       'NAMAKKAL':11,
                       'PERAMBALUR':12,
                       'PUDUKKOTTAI':13,
                       'RAMANATHAPURAM':14,
                       'SALEM': 15,
                       'SIVAGANGA': 16,
                       'THANJAVUR': 17,
                       'THE NILGIRIS': 18,
                       'THENI': 19,
                       'THIRUVALLUR': 20,
                       'THOOTHUKUDI': 21,
                       'TIRUCHIRAPPALLI': 22,
                       'TIRUNELVELI': 23,
                       'TIRUPPUR': 24,
                       'TIRUVANNAMALAI': 25,
                       'VELLORE': 26,
                       'VILLUPURAM': 27,
                       'VIRUDHUNAGAR': 28})


warnings.filterwarnings(action='ignore',
                        category=DeprecationWarning,
                        module='Numpy')

from sklearn.model_selection import train_test_split

df.drop(df.columns[np.isnan(df).any()], axis=1)
X = df.drop(columns='PLACE')
y = df['PLACE']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)




X_train = np.nan_to_num(X_train)

y_train = np.nan_to_num(y_train)

X_test = np.nan_to_num(X_test)

y_test = np.nan_to_num(y_test)

warnings.filterwarnings(action='ignore',
                        category=DeprecationWarning,
                        module='Numpy')

from sklearn.svm import SVC
from sklearn.metrics import classification_report
svc = SVC( random_state=0)
svc.fit(X_train, y_train)




y_pred = svc.predict(X_test)
print(classification_report(y_test, y_pred))

print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test, y_test)))

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))

