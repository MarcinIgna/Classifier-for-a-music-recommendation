import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree
music_data = pd.read_csv("music.csv")
# X = music_data.drop(columns=['genre'])# input dataset drop deleting column
y = music_data['genre']# output dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)# 20% of data for testing
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)

# joblib.dump(model, 'music-recommender.joblib')# save model
model = joblib.load('music-recommender.joblib')# loading model
predictions = model.predict([[21, 1]])# predict
tree.export_graphviz(model, out_file='music-recommender.dot',
                    feature_names=["age", "gender"], # rules for each node
                    class_names=sorted(y.unique()), # class for each node
                    label='all',  # labels for each node
                    rounded=True,  # rounded corners
                    filled=True)  # filtered with colour

# score = accuracy_score(y_test, predictions)# it will give accuracy score from 0 to 1
# print(score)

