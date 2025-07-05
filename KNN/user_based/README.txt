KNN-based recommendation system that recommends movies to users based on their preferences and similar users’ choices.
Data set canbe obtained from https://grouplens.org/datasets/movielens/

This is an example of KNN User based Filtering -
Find users similar to the current user, then recommend movies those similar users liked.
------------------------------------------------------------------------------------------------------

Scenario:
There are 10 users who rate food items like Maggie, Pasta, Pizza, etc.

You’re one of the users: User A

You rated:

✅ Maggie → 5 stars

✅ Pasta → 4 stars

❌ Pizza → 1 star

✅ Now, User-Based Filtering does this:
"Let me find other users who also liked Maggie and Pasta, and disliked Pizza just like you."

Let’s say it finds:
User B has very similar tastes to you (also loves Maggie + Pasta, dislikes Pizza)
User B has also rated Noodles highly
➡ So the system concludes:
“Since User B is similar to you, and they liked Noodles, we’ll recommend Noodles to you too.”

