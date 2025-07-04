KNN-based recommendation system that recommends movies to users based on their preferences and similar users’ choices.
Data set canbe obtained from https://grouplens.org/datasets/movielens/

Two Perspectives in Item-Based Filtering

Phase 1: Build the item-item similarity matrix -- cosine_similarity calculation.
How do we know Inception is similar to Interstellar?

➡ We look at how 1,000 users rated both:

If users who liked Inception also liked Interstellar → they're similar.
This is global, computed once using other users’ rating patterns.
similarity_matrix = cosine_similarity(movie_user_matrix)
So yes — "other users" define the similarity between movies.

🎯 Phase 2: Recommend based on what user X liked
Now a new user comes in and rates Inception highly.

➡ We look at the movies that user X liked (e.g., Inception)
➡ Then recommend movies similar to those liked movies — i.e., Interstellar.

This is personalized using that user's history.

✅ So the final explanation is:
“We build item similarity using other users’ past ratings,
then for a specific user, we recommend items similar to the items they liked.”

✨ One-Line Summary for Item-Based:
"Use crowd behavior to learn how items are similar, then recommend similar items to what the user liked."

----------------------------------------------------------------------------------------------------------------------------------------

cosine_similarity is exactly what checks the rating for other users.
How other users have rated the Movies.

Let’s say we have:
3 users: U1, U2, U3
2 movies: A and B
And the rating matrix looks like:
User	Movie A	Movie B
U1	5	5
U2	4	4
U3	2	1

Step: Cosine Similarity Between Movie A & B
We treat each movie as a vector of user ratings:

Movie A → [5, 4, 2]

Movie B → [5, 4, 1]

cosine_similarity([MovieA_vector], [MovieB_vector])
This compares how similarly the two movies are rated across all users.
So yes — this is where “other users” affect the model.
------------------------------------------------



 1. “Other users ka kya perception hai kisi item ke baare mein”
➡ This is how item similarity is built.

If most users rated Inception and Interstellar similarly (both highly), then these two movies are considered similar.
So yes — it captures collective perception of items.

 2. “Maine pehle kya pasand kiya tha”
➡ This is how your personal recommendations are made.

If you rated Inception highly,
And the model knows Interstellar is similar to Inception (from collective data),
Then it recommends Interstellar to you.

✅ This is personalization based on your own past history








