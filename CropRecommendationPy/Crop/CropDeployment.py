# Importing essential libraries
import numpy as np
import pandas as pd
import pickle
import warnings

import warnings
warnings.filterwarnings('ignore')

# Loading the dataset
df = pd.read_csv('./Crop_recommendation.csv')


'''def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


df = clean_dataset(df)'''


#import pandas as pd
import matplotlib.pyplot as plt

# read-in data
#data = pd.read_csv('./test.csv', sep='\t') #adjust sep to your needs

import seaborn as sns
sns.countplot(df['label'],label="Count")
plt.show()


df.label=df.label.map({'rice':0,
                       'maize':1,
                       'chickpea':2,
                       'kidneybeans':3,
                       'pigeonpeas':4,
                       'mothbeans':5,
                       'mungbean':6,
                       'blackgram':7,
                       'lentil':8,
                       'pomegranate':9,
                       'banana':10,
                       'mango':11,
                       'grapes':12,
                       'watermelon':13,
                       'muskmelon':14,
                       'apple': 15,
                       'orange': 16,
                       'papaya': 17,
                       'coconut': 18,
                       'cotton': 19,
                       'jute': 20,
                       'coffee': 21})

# Replacing the 0 values by NaN
df_copy = df.copy(deep=True)
df_copy[['N','P','K','temperature','humidity','ph','rainfall']] = df_copy[['N','P','K','temperature','humidity','ph','rainfall']].replace(0,np.NaN)





# Model Building
from sklearn.model_selection import train_test_split
df.drop(df.columns[np.isnan(df).any()], axis=1)
X = df.drop(columns='label')
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)




from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression().fit(X_train, y_train)
print(" LR Accuracy on training set: {:.3f}".format(logreg.score(X_train, y_train)))
print(" LR Accuracy on test set:  {:.3f}".format(logreg.score(X_test, y_test)))







from sklearn.svm import SVC
classifier = SVC(random_state=0)
classifier.fit(X_train, y_train)


print(" SVM Accuracy on training set: {:.2f}".format(classifier.score(X_train, y_train)))
print(" SVM Accuracy on test set: {:.3f}".format(classifier.score(X_test, y_test)))




from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("DTC Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("DTC Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))



from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)

print("RandomForestClassifier Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
print("RandomForestClassifier Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))








# Creating a pickle file for the classifier
'''filename = 'crop-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

classifier = pickle.load(open(filename, 'rb'))

data = np.array([[73,43,42,26.58361011,78.00774772,6.310699968,154.8238864]])
my_prediction = classifier.predict(data)

warnings.filterwarnings("ignore", category=DeprecationWarning)
print(my_prediction)

if my_prediction == 0:
    Answer = 'rice'

    print(Answer)

else:
    Answer = 'No-Crop-Found'
   
    print(Answer)'''



