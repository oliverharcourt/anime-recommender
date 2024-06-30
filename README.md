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
