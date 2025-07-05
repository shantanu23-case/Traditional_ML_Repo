from sklearn.tree import DecisionTreeClassifier
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


##-->Hum average rating har movie ka nikalenegey plus count 
#krengey kitne logon ne rate kia h

movie_features = movies_rating.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
# This created a DataFrame like:
# movieId	mean	count
# 1	        4.2	    1123
# 2	        3.7	    984

movie_features.columns = ['moviesID','avg_Rating','rating_count']
#We are jsut renaming the coloums
#But we still include it in the list to match all 3 column names at once.
##-->Merge kr rae hai movies feature mein movies.

movie_features=movie_features.merge(movies[['movieId', 'genres']], on='movieId')
movie_features['main_genre'] = movie_features['genres'].apply(lambda x: x.split('|')[0])

# movies[['movieId', 'genres']]
# This selects just movieId and genres columns from the movies DataFrame.
# on='movieId'
# This tells Pandas:
# In Your Case:
# 1. movie_features (from ratings):
# movieId	avg_rating	rating_count
# 1	4.2	2000
# 2	3.8	1500

# 2. movies[['movieId', 'genres']] (from movie metadata):
# movieId	genres
# 1	Adventure
# 2	Drama

# After this line:
# movie_features = movie_features.merge(movies[['movieId', 'genres']], on='movieId')
# You get:
# movieId	avg_rating	rating_count	genres
# 1	4.2	2000	Adventure
# 2	3.8	1500	Drama

le = LabelEncoder()
movie_features['genre_label'] = le.fit_transform(movie_features['main_genre'])
# A new column genre_label is created, which contains numeric versions of genres.
# These numbers are needed because:
# ML models like DecisionTreeClassifier or RandomForestClassifier can’t handle text directly.
# They only work with numeric values.
#le = LabelEncoder()
#this line (written earlier) creates a LabelEncoder object from scikit-learn — a tool that converts text labels to numbers.

#movie_features['main_genre']
# This column contains genres in text format, like:
# main_genre
# Action
# Comedy
# Drama
# Action

# le.fit_transform(...)
# This does two things in one step:
# .fit() learns all unique genres (e.g. "Action", "Comedy", "Drama")
# .transform() converts them into numbers like:
# Action → 0
# Comedy → 1
# Drama → 2


# Step 3: Split data into features and labels
X = movie_features[['avg_rating', 'rating_count']]
y = movie_features['genre_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 4: Train Decision Tree
dtree = DecisionTreeClassifier(max_depth=5, random_state=42)
dtree.fit(X_train, y_train)

# Predict on test data
y_pred = dtree.predict(X_test)
# Print accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))