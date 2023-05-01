import pickle

import numpy as np
import pandas as pd
import re


def load_data():
    """
    Load Dataset from File
    """
    # 读取User数据
    users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users = pd.read_csv('dataset/users.dat', sep='::', header=None, names=users_title, engine='python')
    users = users.filter(regex='UserID|Gender|Age|JobID')
    users_orig = users.values
    # 改变User数据中性别和年龄
    gender_map = {'F': 0, 'M': 1}
    users['Gender'] = users['Gender'].map(gender_map)

    # enumerate迭代的顺序不是按照数值。可能按照存储位置。
    age_map = {val: ii for ii, val in enumerate(set(users['Age']))}
    users['Age'] = users['Age'].map(age_map)

    # 读取Movie数据集
    movies_title = ['MovieID', 'Year', 'Genres']
    movies = pd.read_csv('dataset/movies.dat', sep='::', header=None, names=movies_title, engine='python')
    movies = movies.filter(regex='MovieID|Year|Genres')
    movies_orig = movies.values

    # 年份转化为数字字典
    pattern = re.compile(r'^(.*)\((\d+)\)$')
    year_map = {val: pattern.match(val).group(2) for ii, val in enumerate(set(movies['Year']))}
    movies['Year'] = movies['Year'].map(year_map)
    year_map = {val: ii for ii, val in enumerate(set(movies['Year']))}
    movies['Year'] = movies['Year'].map(year_map)

    # 电影类型转数字字典
    genres_set = set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)
    # PAD占位符，目的为将向量补全为等长以输入到网络。
    genres_set.add('<PAD>')
    genres2int = {val: ii for ii, val in enumerate(genres_set)}
    # 将电影类型转成等长数字列表，长度是26(movie数据集是18)
    genres_map = {val: [genres2int[row] for row in val.split('|')] for ii, val in enumerate(set(movies['Genres']))}
    # 将转化成功的向量之后添加占位符进行等长的补全。
    for key in genres_map:
        for cnt in range(max(genres2int.values()) - len(genres_map[key])):
            genres_map[key].insert(len(genres_map[key]) + cnt, genres2int['<PAD>'])
    movies['Genres'] = movies['Genres'].map(genres_map)

    # 读取评分数据集
    ratings_title = ['UserID', 'MovieID', 'Rating', 'timestamps']
    ratings = pd.read_csv('dataset/ratings.dat', sep='::', header=None, names=ratings_title, engine='python')
    ratings.sort_values(by=['UserID', 'timestamps'],inplace=True)
    ratings = ratings.filter(regex='UserID|MovieID|Rating')

    ratings.loc[ratings[ratings.Rating <= 3].index.tolist(),'Rating'] = 0
    # ratings.drop(index=ratings[ratings.Rating == 3].index.tolist(),inplace=True)
    ratings.loc[ratings[ratings.Rating > 3].index.tolist(),'Rating'] = 1

    # 合并三个表
    data = pd.merge(pd.merge(ratings, users), movies)

    # 将数据分成X和y两张表
    target_fields = ['Rating']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]

    # 列表（向量）形式。
    features = features_pd.values
    targets_values = targets_pd.values

    # 返回值：类型转化为Int，输入特征，输出目标值特征， 评分数据 ，用户数据，电影数据，合并数据，电影特征，    用户特征...
    return genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig

if __name__ == '__main__':
    genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = load_data()
    pickle.dump((genres2int, features, targets_values, ratings, users, movies, data, movies_orig,
                users_orig), open('preprocess_CTR.p', 'wb'))
    pass