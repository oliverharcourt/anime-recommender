# Anime Recommender

## Plan

1. model takes a user id and gets their completed and watching list.
2. model gets embeddings or generates embeddings (depends if embeddings for anime are already known) for all animes pulled from user
3. calculate the k closest animes in the embedding space
4. distance calculation is weighted based on user ratings of animes in their lists e.g.:
   1. user rated: anime A -> 10, B -> 7, C -> 3
   2. the distance calculation weights higher rated anime more
   3. e.g. weights in order: w(A) > w(B) > w(C)
   4. thus recommendations generated are closer to anime A
5. model returns the k closest anime according to weighted distance metrics
