import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

#merge csv file
df1 = pd.read_csv('ratings.csv')
df2 = pd.read_csv('movies.csv')
links_df = pd.read_csv('links.csv')
tags_df = pd.read_csv('tags.csv')
df_clean = df1.merge(df2, on='movieId')
del df1 
del df2



#Remove users who rated less than 5 movies and movies rated by less than 2 users
# Count the number of ratings per user and movie
user_ratings_count = df_clean['userId'].value_counts()
movie_ratings_count = df_clean['movieId'].value_counts()

# Filter out users who rated less than 5 movies
users_to_keep = user_ratings_count[user_ratings_count >= 35].index
df_filtered_users = df_clean[df_clean['userId'].isin(users_to_keep)]

# Filter out movies rated by less than 2 users
movies_to_keep = movie_ratings_count[movie_ratings_count >= 12].index
df_filtered = df_filtered_users[df_filtered_users['movieId'].isin(movies_to_keep)]

# del df_filtered_users , users_to_keep , movies_to_keep , user_ratings_count , movie_ratings_count
del movie_ratings_count
del user_ratings_count
del df_filtered_users
del users_to_keep
del movies_to_keep


final_dataset = df_filtered.pivot(index='movieId',columns='userId',values='rating')


# replace NaN with 0
final_dataset.fillna(0,inplace=True)



csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)



knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)



def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 10
    movie_list = df_filtered[df_filtered['title'].str.contains(movie_name, case=False)]  # Use case-insensitive matching
    
    if not movie_list.empty: 
        movie_idx = movie_list.iloc[0]['movieId']  # Get the movieID
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]  # Get the corresponding movieID in the final dataset
        
        # Get the distance and index of the nearest top 10 movies
        distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_reccomend+1)
        
        # Sort movies according to distance ascending (closest first)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[1:]  
        
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = df_filtered[df_filtered['movieId'] == movie_idx].index
            
            if not idx.empty:  # Ensure the index exists
                title = df_filtered.iloc[idx[0]]['title']
                distance = val[1]
                
                # Get the link associated with the movie
                link = links_df[links_df['movieId'] == movie_idx]['imdbId'].values
                link = f"https://www.imdb.com/title/tt{int(link[0]):07d}/" if len(link) > 0 else 'N/A'
                
                # Get the tags associated with the movie
                tags = tags_df[tags_df['movieId'] == movie_idx]['tag'].values
                tags = ', '.join(tags) if len(tags) > 0 else 'No tags available'
                
                recommend_frame.append({
                    'Title': title, 
                    'Distance': distance,
                    'Link': link,
                    'Tags': tags
                })
        
        df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_reccomend + 1))
        return df
    else:
        return "No movies found. Please check your input"


get_movie_recommendation('Rambo')