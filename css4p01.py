# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:05:31 2024

@author: 33553904
"""

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv("movie_dataset.csv")

# Look at histograms to get idea of how variables are distrubuted (overall)

df.hist(color='DarkBlue',figsize= (10,10))

# Explore general information

print(df)

df.describe()

df.info()

df.head()

# Drop duplicates

df.drop_duplicates(inplace=True)

# View missing values

print(df.isnull().sum())

print("Dataframe has null values?", df.isnull().values.any())

# Replace NaNs with 0

df['Revenue (Millions)'].fillna(0, inplace=True)

df['Metascore'].fillna(0, inplace=True)

# Check if missing values are removed

print(df.isnull().sum())
print("Dataframe has null values?", df.isnull().values.any())



#Question 1: What is the highest rated movie in the dataset?

print(df.nlargest(1, 'Rating'))

#Question 2: What is the average revenue of all movies in the dataset?

print(df["Revenue (Millions)"].mean())

#Question 3: What is the average revenue of movies from 2015 to 2017 in the dataset?

df2 = df.query("`Year` >= 2015 and `Year` <= 2017")
print(df2["Revenue (Millions)"].mean())

#Question 4: How many movies were released in the year 2016?

df3=df.query("Year == 2016")
print(len(pd.unique(df3['Title'])))

#Question 5: 
    
df4 = df.query("Director == 'Christopher Nolan'")
print(len(pd.unique(df4['Title'])))

#Question 6: How many movies in the dataset have a rating of at least 8.0?

df5 = df.query("`Rating` >= 8.0")
print(len(pd.unique(df5['Title'])))

#Question 7: What is the median rating of movies directed by Christopher Nolan?

print(df4["Rating"].mean())

#Question 8: Find the year with the highest average rating?

df6 = df.groupby(df.Year)['Rating'].mean()

#Question 9:What is the percentage increase in number of movies made between 2006 and 2016?

df7 = df.query("Year == 2006")
fin = len(pd.unique(df3['Title']))
init = len(pd.unique(df7['Title']))
percent_increace = ((fin - init)/init)*100
print(percent_increace)

#Question 10: Find the most common actor in all the movies?
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sen = df['Actors'].str.split(',').apply(Series, 1).stack()
sen.index = sen.index.droplevel(-1)
sen.name = 'Actors1'
del df['Actors']
df8 = df.join(sen)
df8['Actors1'].value_counts().nlargest(2)
print(df8['Actors1'].value_counts().nlargest(2))

#Question 11: How many unique genres are there in the dataset?
s = df['Genre'].str.split(',').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'Genres'
del df['Genre']
df9 = df.join(s)
df9['Genres'].unique()
print(df9['Genres'].unique())
len(df9['Genres'].unique())
print(len(df9['Genres'].unique()))

#Question 12: Do a correlation of the numerical features, what insights can you deduce? Mention at least 5 insights.
print(df.nlargest(1, 'Rating'))
df = pd.read_csv("movie_dataset.csv")
plt.scatter(df.loc[:,"Rating"],df.loc[:,"Revenue (Millions)"])
plt.title('Rating vs Revenue')
plt.ylabel('Rating')
plt.xlabel('Revenue')
plt.show()


s = df['Genre'].str.split(',').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'Genres'
del df['Genre']
df9 = df.join(s)
df9['Genres'].unique()
print(df9['Genres'].unique())
len(df9['Genres'].unique())
print(len(df9['Genres'].unique()))

genres = df9['Genres'].value_counts()[df9['Genres'].value_counts() > 10].index
genres
gn_data=df9[df9['Genres'].isin(genres)].groupby('Genres')['Revenue (Millions)'].sum().plot(kind='bar', title='Revenue and Genre')
plt.xlabel('Genres')
plt.ylabel('Revenue (Millions)')
plt.show()
gn_data

df[['Rating','Votes']].groupby('Rating').count().plot(kind='bar', title='Rating count')
plt.xlabel('Rating')
plt.ylabel('Votes')
plt.show()

df[['Rating','Metascore']].groupby('Rating').count().plot(kind='bar', title='Rating count vs Metascore')
plt.xlabel('Metascore')
plt.ylabel('Rating')
plt.show()
fig, ax = plt.subplots(figsize=(25,10))
df.groupby('Year').count()['Title'].plot(kind='bar',fontsize=13,color="#2E86C1")
plt.title("Number of movies released per year", fontsize=18)
plt.xlabel("Years (2006-2016)", fontsize=16)
plt.ylabel("Number of movies released", fontsize=16)
plt.show()