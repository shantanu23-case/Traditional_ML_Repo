1: Data Load Karo
📌 ratings file mein users ke rating details hain
📌 movies file mein movie title aur genre info hai


2: Movie ke liye Feature Nikaalo
movie_features = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
movie_features.columns = ['movieId', 'avg_rating', 'rating_count']
📌 Har movie ka:
Average rating
Kitne logon ne rating di — ye dono calculate ho gaya

3. Genre Add Karo Movie Features Mein
movie_features = movie_features.merge(movies[['movieId', 'genres']], on='movieId')
movie_features['main_genre'] = movie_features['genres'].apply(lambda x: x.split('|')[0])
📌 Hum movies ke saath genre bhi jod rahe hain (pehla genre select kar rahe hain)

4. Genre Ko Encode Karo (Text ➝ Number)
le = LabelEncoder()
movie_features['genre_label'] = le.fit_transform(movie_features['main_genre'])
📌 ML model ko genre text samajh nahi aata, isliye number mein convert kar diya


5.  X aur y Prepare Karo
X = movie_features[['avg_rating', 'rating_count']]  # Input features
y = movie_features['genre_label']  # Output label (genre)
📌 X: features jisse prediction hoga
📌 y: actual genre (label)

6: Data Split Karo (Training & Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
📌 80% training ke liye, 20% testing ke liye

7. Decision Tree Model Train Karo
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
📌 Model training complete ho gaya — ab wo seekh gaya genre predict karna

8. Predict Karo aur Accuracy Check Karo
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
📌 Ye batayega ki model ne kitne sahi genre predict kiye

🔹 Step 10: Nayi Movie ke liye Genre Predict Karo
python
Copy
Edit
sample = pd.DataFrame([[4.3, 800]], columns=['avg_rating', 'rating_count'])
predicted_label = model.predict(sample)
predicted_genre = le.inverse_transform(predicted_label)
print("Predicted Genre:", predicted_genre[0])
📌 Yahan aap test kar rahe ho:

Agar koi movie ka rating 4.3 hai aur 800 logon ne rate kiya, to wo kis genre ki ho sakti hai?

✅ Final Thoughts
Step	What You Did
Data Load	Rating & Genre data use kiya
Feature Prep	Avg rating, rating count banaya
Label Encode	Genre ko numbers mein convert kiya
Model Train	Decision Tree use kiya
Predict & Evaluate	Accuracy check kiya & test movie pe prediction kiya