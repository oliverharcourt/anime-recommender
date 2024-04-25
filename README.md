# Anime Recommender

## Plan

### Approach 1

1. model takes a user id and gets their completed and watching list
2. model gets embeddings or generates embeddings (depends if embeddings for anime are already known) for all animes pulled from user
3. calculate the k closest animes in the embedding space
4. distance calculation is weighted based on user ratings of animes in their lists e.g.:
   1. user rated: anime A -> 10, B -> 7, C -> 3
   2. the distance calculation weights higher rated anime more
   3. e.g. weights in order: w(A) > w(B) > w(C)
   4. thus recommendations generated are closer to anime A
5. model returns the k closest anime according to weighted distance metrics


### Approach 2

1. model takes a user id and gets their completed and watching list
2. model gets embeddings or generates embeddings (depends if embeddings for anime are already known) for all animes pulled from user
3. calculate variances for gaussian distributions centered around the user animes in the embedding space based on:
   1. user rated: anime A -> 10, B -> 7, C -> 3
   2. the gaussian distribution centered around anime A should get a greater variance
   3. e.g. weights in order: Var of N around A > Var of N around B > Var of N around C
   4. (also somehow make sure prob of animes nearer to A is higher than prob of animes near C)
   5. thus recommendations have higher chance of being close to A, but can also be slightly further from A than just the closest anime. this could help with the filter bubble effect, where all recommendations are essentially of the same kind of anime and the user is never introduced to new anime.
4. model returns the k closest anime according to weighted distance metrics
