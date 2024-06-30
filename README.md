# Anime Recommender

## General Info

This project is a simple recommendation engine for MyAnimeList. It uses Milvus to search for similar animes to the ones in a users anime list.

## Usage

To use this project to generate anime recommendations for a given MAL account, simply run:

```shell
$ python main.py -u [user_name]
```

To generate recommendations based on a single anime, run:

```shell
$ python main.py -a [anime_title]
```

The script uses a fuzzy search over the anime dataset to find the animes matching the search string. This process will ask the user to select the correct anime.

The ``l`` flag can be used to specify the desired number of recommendations to generate.

## Examples

```shell
$ python main.py -a "One"

The following animes were found:
1. One Piece
2. JoJo no Kimyou na Bouken Part 6: Stone Ocean Part 3
3. Koukaku Kidoutai: Stand Alone Complex 2nd GIG
4. One Punch Man
5. Koukaku Kidoutai: Stand Alone Complex
Select the anime you are looking for, or n to abort search (1/2/.../n): 1
Recommendations for anime with id: 21
      id  distance                                 link
0  19123       0.7  https://myanimelist.net/anime/19123
1  38234       0.7  https://myanimelist.net/anime/38234
2    136       0.6    https://myanimelist.net/anime/136
3    137       0.6    https://myanimelist.net/anime/137
4    138       0.6    https://myanimelist.net/anime/138
5    139       0.6    https://myanimelist.net/anime/139
6  11061       0.6  https://myanimelist.net/anime/11061
7  18115       0.6  https://myanimelist.net/anime/18115
8  41467       0.6  https://myanimelist.net/anime/41467
9  53998       0.6  https://myanimelist.net/anime/53998
```
```shell
$ python main.py -a "Death" -l 15

The following animes were found:
1. Death Note
2. Death Parade
3. Death Billiards
4. Death Note: Rewrite
5. Dead Mount Death Play Part 2
Select the anime you are looking for, or n to abort search (1/2/.../n): 1
Recommendations for anime with id: 1535
       id  distance                                 link
0    2994  0.600000   https://myanimelist.net/anime/2994
1   16762  0.600000  https://myanimelist.net/anime/16762
2   41619  0.300000  https://myanimelist.net/anime/41619
3   35120  0.196532  https://myanimelist.net/anime/35120
4     355  0.196397    https://myanimelist.net/anime/355
5    8247  0.196235   https://myanimelist.net/anime/8247
6   36144  0.196112  https://myanimelist.net/anime/36144
7   21843  0.196070  https://myanimelist.net/anime/21843
8   51417  0.196068  https://myanimelist.net/anime/51417
9    2053  0.196052   https://myanimelist.net/anime/2053
10   2772  0.196012   https://myanimelist.net/anime/2772
11  51367  0.195960  https://myanimelist.net/anime/51367
12  54595  0.195960  https://myanimelist.net/anime/54595
13    905  0.195926    https://myanimelist.net/anime/905
14   2246  0.195890   https://myanimelist.net/anime/2246
```
