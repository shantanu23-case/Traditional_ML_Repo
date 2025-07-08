from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd

##-->Load Data set
movies_rating = pd.read_csv("data/ml-10m/ml-10M100K/ratings.dat",sep="::", 
                            engine="python", 
                            names=["userId", "movieId", "rating", "timestamp"])
movies=pd.read_csv("data/ml-10m/ml-10M100K/movies.dat",
                   sep="::", 
                   engine="python", names=["movieId", "title", "genres"])
#Merge data
data = pd.merge(movies_rating,movies,on='moviesId')

###############################       Predict if a user will like a movie (rating ≥ 4)               ###############################
# Create classification label

#we are doing it so that we can convert rating data for example 2,3,4 to set into binary form like
#if my rating is less than<4 then mark its as 0 if above or eqoalu to 4 then
# mark it 1. 
data['like'] = data['rating'].apply(lambda r: 1 if r >= 4 else 0)
y_clf = data['like']

#encode Genre
data['main_genre'] = data['genres'].apply(lambda x: x.split('|')[0])
le = LabelEncoder()
data=['genre_encoded']= le.fit_transform(data['main_genre'])

#
X=data[['userId','movieId','genre_encoded']]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
#train model
#n_estiamtors means no. of decison trees.Its like an advisor of shivkamni devi from bahubali
#IDEALLY n=100 is a good value when you have good amount of  data set
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(" Classification Accuracy:", accuracy_score(y_test, y_pred))


###############################       Predict the exact rating (rating 3.2)               ###############################

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#we imported the mean square erro so as to evaluate how good your regression model is.
# Because in regression (like RandomForestRegressor), we predict numeric values (e.g., ratings like 3.5, 4.0, etc.).
# To measure how far off our predictions are from the actual values, we use an error metric — and MSE is one of the most common.

y_reg = data['rating']
# Split data
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
model_reg =RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train_r,y_train_r)

y_pred_reg = model_reg.predict(X_test_r)
mse =mean_squared_error(y_test_r,y_pred_reg)
print(mse)

# Classification Accuracy → How well it predicts Like/Dislike
# Regression MSE → How close predicted ratings are to real ones