# import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
import string
import joblib

# Load datasets
fake =(pd.read_csv('/content/fake.csv.zip'))
true =(pd.read_csv('/content/true.csv.zip'))

from google.colab import drive
drive.mount('/content/drive')

# To print fake data
fake.head()

# To print real data
true.head()

#label data
fake['class']=0         # 0-------> fake news
true['class']=1          # 1-------> real news

# Merge datasets
date =pd.concat([fake,true],axis=0)

# drop irrelevant columns
data = date.drop(['text','subject','date'],axis=1)    # Focusing only on 'title'

data.reset_index(inplace=True)

data.drop(['index'],axis=1,inplace=True)

data.sample(5)

# cleaning function
def clear_title(title):
  title =title.lower()
  title = re.sub("\n","",title)
  return title
  # Apply cleaning
  data['title'] = data['title'].apply(clear_title)
  return title

#Vectorization & Training
x = data['title']
y = data['class']
# Split data into 75% training and 25% testing
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.25,random_state=42)

# Convert text titles to numerical data
from numpy import vectorize
vectorize = TfidfVectorizer()
xv_train = vectorize.fit_transform(xtrain)
xv_test = vectorize.transform(xtest)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(xv_train,ytrain)

# Evaluation
predict = model.predict(xv_test)
model.score(xv_test,ytest)
print("Model Accuracy:", model.score(xv_test, ytest))

print(classification_report(ytest,predict))

# Save the model and vectorizer for the UI
joblib.dump(vectorize,'vectorize.joblib')
joblib.dump(model,'model.joblib')
