import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
import scipy
import sklearn.neighbors as spn
import turicreate
import implicit
import random
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from collections import defaultdict

# dataset file names
full_user_profile_file = 'usersha1-profile.tsv'
listening_history_file = 'usersha1-artmbid-artname-plays.tsv'
working_directory = os.getcwd()

# dataset headers
user_profile_headers = ['user_id', 'user_gender', 'user_age', 'user_country', 'user_registration_date']
user_history_headers = ['user_id', 'artist_id', 'artist_name', 'play_count']
# rows to skip or drop
unique_to_drop = ['user_id', 'artist_name']
user_data_delete = ['user_gender', 'user_registration_date', 'user_age']
listening_data_delete = ['artist_id']

# spare matrix file_name
sparse_file_name = 'generated_sparse.npz'

# KNN Neighbor count for model
knn_count = 5


def load_listening_data(file_name):
    '''Cleands and loads the dataset.

    :param file_name: string, dataset filename
    :return: pandas dataframe
    '''
    # read the csv file into the dataframe, skipping bad rows
    file_path = '{}/{}'.format(working_directory, file_name)
    recommender_testing(file_path)

    df = pd.read_csv(file_path, header=None, sep='\t', error_bad_lines=False)
    df.columns = user_history_headers

    # delete unnecessary columns
    for col in listening_data_delete:
        del df[col]

    # drop rows that don't have both a user_id and song_id
    df.dropna(subset=['artist_name'], inplace=True)

    # group data by artist_name and play_count
    grouped_data = df.groupby(by='artist_name')['play_count']
    # get the artist play count
    artists_play_count = grouped_data.sum().reset_index()
    all_play_counts = (artists_play_count[['artist_name', 'play_count']])
    print(all_play_counts.head())
    # merge artist play count dataframe
    merge_key = 'artist_name'
    artist_plays_combined = df.merge(all_play_counts, left_on=merge_key, right_on=merge_key, how='left')
    # only consider artists with more than 10k plays
    popular_artists = artist_plays_combined.query('play_count_y >= 10000')
    # rename column names
    popular_artists = popular_artists.rename(
        columns={'play_count_x': 'user_play_count', 'play_count_y': 'total_play_count'})
    print(popular_artists.head())
    print('popular artists dataset built, total_play_count description below')
    print(popular_artists['total_play_count'].describe())
    # generate and save various plots for our artis/user play data
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.rcParams.update({'figure.autolayout': True})
    listening_sample = popular_artists.sample(10)

    # plot bar plots of total play counts and user play counts
    fig = listening_sample[['total_play_count']].plot(kind='bar')
    fig.set_xlabel("Artist Name")
    fig.set_ylabel("Total Artist plays")
    fig.set_title("Artist Name vs Total PLays")
    fig.set_xticklabels(listening_sample.artist_name)
    fig = fig.get_figure()
    fig.savefig('random_artists_total_plays.png')

    fig = listening_sample[['user_play_count']].plot(kind='bar')
    fig.set_xlabel("Artist Name")
    fig.set_ylabel("User Artist plays")
    fig.set_title("Artist Name vs User PLays for Artist")
    fig.set_xticklabels(listening_sample.artist_name)
    fig = fig.get_figure()
    fig.savefig('random_artists_user_plays.png')

    print('images generated')
    return popular_artists


def load_user_data(file_name, popular_artists):
    '''
    
    :param file_name: 
    :param artist_play_counts: 
    :return: 
    '''

    # read the csv file into the dataframe, skipping bad rows
    file_path = '{}/{}'.format(working_directory, file_name)
    df = pd.read_csv(file_path, header=None, sep='\t', error_bad_lines=False)
    df.columns = user_profile_headers

    # delete unnecessary columns
    for col in user_data_delete:
        del df[col]

    # plot fig (takes long)
    # fig = df[['user_age', 'user_country']].plot(kind='bar')
    # print('user age plot fig created')
    # fig.set_xlabel("User Country")
    # fig.set_ylabel("User Age")
    # fig.set_title("Artist Name vs User PLays for Artist")
    # fig.set_xticklabels(df.user_country)
    # fig = fig.get_figure()
    # fig.savefig('user_age_vs_country.png')
    # print('plot saved')
    # del df['user_age']

    # join popular artists with users
    merge_key = 'user_id'
    joined_popular = popular_artists.merge(df, left_on=merge_key, right_on=merge_key, how='left')
    # limit to only US and UK users
    # selected_users = joined_popular[(joined_popular.user_country == 'United States') | (joined_popular.user_country == 'United Kingdom')]
    selected_users = joined_popular[(joined_popular.user_country == 'United States')]

    # drop all duplicated users/artist pairs in our selection
    selected_users = selected_users.drop_duplicates(unique_to_drop)
    # returns US and UK users with their play counts for a song
    print('selected users dataset built')

    return selected_users


def build_utility_matrix(dataframe, matrix_row_param, matrix_column_param, val_param):
    '''builds an efficient utility matrix for the songs and users 
    
    :param dataframe: 
    :param matrix_row_param: string, row field in utility matrix
    :param matrix_column_param: string, column field in utility matrix
    :return: utility matrix as scipy sparse csr_matrix
    '''
    # build the utility matrix
    # utility = pd.DataFrame(index=matrix_index, columns=matrix_columns)
    print('started utility build')
    utility_filled = dataframe.pivot(index=matrix_row_param, columns=matrix_column_param, values=val_param).fillna(0)
    # utility_filled = utility.fillna(0)
    # del utility
    print('deleted utility')
    # convert to a more memory efficient matrix
    sparse_utility = utility_filled.apply(np.sign)
    print('sparse utility values converted')
    sparse_utility_csr = sp.csr_matrix(sparse_utility.values)
    print('sparse utility matrix created')
    # save generated sparse matrix
    file_path = '{}/{}'.format(working_directory, sparse_file_name)
    print('saved sparse file')
    sp.save_npz(file_path, sparse_utility_csr)
    return sparse_utility


def build_model(matrix_file_path, metric_type):
    '''Build a Nearest Neighbor model using scikit and fit it to the dataset.
    
    :param matrix_file_path: path to matrix, generated artist, user pair utility matrix
    :param metric_type: metric to measure artist similarity
    :return: datafit scikit nearest neighbor model
    '''
    # open saved utility matrix (done to save time on reruns)
    utility_matrix = sp.load_npz(matrix_file_path)
    print('loaded utility matrix in')
    # build model object
    nearest_neighbor_model = spn.NearestNeighbors(algorithm='brute', metric=metric_type)
    # fit model to our utility matrix
    nearest_neighbor_model.fit(utility_matrix)
    print('model built and fit to data')
    return nearest_neighbor_model


def make_recommendations_random(matrix_file_path, user_artist_data, model):
    '''Uses the model to make recommendations based on a given artist name.
    
    :param matrix_file_path: matrix_file_path: path to matrix, generated artist, user pair utility matrix 
    :param user_artist_data: user, artist dataframe
    :param model: Nearest Neighbor model
    :param artist_name: Artist to make recommendations for
    :return: 
    '''
    print('\n Random Recommendations \n')
    # open saved utility matrix (done to save time on reruns)
    # utility_matrix = sp.load_npz(matrix_file_path)
    print('loaded utility matrix in')

    for i in range(1, 10):
        song_position = np.random.choice(user_artist_data.shape[0])
        # reshape our data
        cosine_dis, positions = model.kneighbors(user_artist_data.iloc[song_position, :].values.reshape(1, -1),
                                                 n_neighbors=knn_count)
        flattened_distances = cosine_dis.flatten()
        print('Recommendations for Artist: {}'.format(user_artist_data.index[song_position]))
        for i in range(1, len(flattened_distances)):
            print('Match {} - {} - cosine distance {}'.format(i, user_artist_data.index[positions.flatten()[i]],
                                                              cosine_dis.flatten()[i]))


def make_recommendations_artists(user_artist_data, model, match_artists):
    '''Makes recommendations on a list of artists
    
    :param user_artist_data: user, artist dataframe
    :param model: Nearest Neighbor model
    :param match_artists: artists to get recommendations for
    :return: 
    '''
    print('\n Specific Artists Recommendations \n')
    # open saved utility matrix (done to save time on reruns)
    # utility_matrix = sp.load_npz(matrix_file_path)
    # print('loaded utility matrix in')

    for ind in user_artist_data.index:
        if ind.lower() in match_artists:
            # get the index of the match
            song_position = user_artist_data.index.tolist().index(ind)
            # reshape our data
            cosine_dis, positions = model.kneighbors(user_artist_data.iloc[song_position, :].values.reshape(1, -1),
                                                     n_neighbors=knn_count)
            flattened_distances = cosine_dis.flatten()
            print('Recommendations for Artist: {}'.format(user_artist_data.index[song_position]))
            for i in range(1, len(flattened_distances)):
                print('Match {} - {} - cosine distance {}'.format(i, user_artist_data.index[positions.flatten()[i]],
                                                                  cosine_dis.flatten()[i]))


def surprise_testing(music_data):
    '''Conducts the testing on our recommender system
    
    :param music_data: 
    :return: 
    '''
    # Code from Surprise documentation
    algo = SVD()
    reader = Reader(rating_scale=(1, 50))
    sampled_data = music_data.sample(500000)
    surprise_data = Dataset.load_from_df(sampled_data[['user_id', 'artist_id', 'plays']], reader)
    trainset = surprise_data.build_full_trainset()
    algo.fit(trainset=trainset)
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    top_n = get_top_n(predictions, n=10)

    # Print the recommended items for each user
    for uid, user_ratings in top_n.items():
        print(uid, [iid for (iid, _) in user_ratings])

    cross_validate(algo, surprise_data, ['RMSE', 'MAE'], cv=4, verbose=True)


def recommender_testing(file_name):
    '''Perform testing on the recommender, main function
    
    :param file_name: dataset file name
    :return: 
    '''
    print('began testing')
    listening_data = pd.read_table(file_name)

    raw_data = listening_data.drop(listening_data.columns[1], axis=1)
    raw_data.columns = ['user', 'artist', 'plays']

    # Drop NaN columns
    data = raw_data.dropna()

    data = data.copy()

    # Create a numeric user_id and artist_id column
    data['user'] = data['user'].astype("category")
    data['artist'] = data['artist'].astype("category")
    data['user_id'] = data['user'].cat.codes
    data['artist_id'] = data['artist'].cat.codes

    # from Surprise documentation
    algo = SVD()
    reader = Reader(rating_scale=(1, 50))
    sampled_data = data.sample(500000)
    surprise_data = Dataset.load_from_df(sampled_data[['user_id', 'artist_id', 'plays']], reader)
    trainset = surprise_data.build_full_trainset()
    algo.fit(trainset=trainset)
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    top_n = get_top_n(predictions, n=10)

    # Print the recommended items for each user
    for uid, user_ratings in top_n.items():
        print(uid, [iid for (iid, _) in user_ratings])

    cross_validate(algo, surprise_data, ['RMSE', 'MAE'], cv=4, verbose=True)

    # Create a numeric user_id and artist_id column
    data['user'] = data['user'].astype("category")
    data['artist'] = data['artist'].astype("category")
    data['user_id'] = data['user'].cat.codes
    data['artist_id'] = data['artist'].cat.codes

    # The implicit library expects data as a item-user matrix so we
    # create two matricies, one for fitting the model (item-user)
    # and one for recommendations (user-item)
    sparse_item_user = scipy.sparse.csr_matrix((data['plays'].astype(float), (data['artist_id'], data['user_id'])))
    sparse_user_item = scipy.sparse.csr_matrix((data['plays'].astype(float), (data['user_id'], data['artist_id'])))

    # Initialize the als model and fit it using the sparse item-user matrix
    model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)

    data_conf = (sparse_item_user).astype('double')
    model.fit(data_conf)
    surprise_testing(surprise_data)




def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''
    # Code from Surprise documentation
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def test_recommender(user_artist_data):
    '''Test the recommender using turicreate
    
    :param user_artist_data:
    :return:
    '''
    # From Turicreate tutorial/documenation
    # create turcreate frame from pandas frame
    print('entered recommender')
    sample = user_artist_data.sample(5)
    del user_artist_data
    tc_frame = turicreate.SFrame(data=sample, format='array')
    print(tc_frame.head())
    print('tc frame created')
    del user_artist_data

    # create the training and test sets by splitting the data randomly
    train_data, test_data = turicreate.recommender.util.random_split_by_user(tc_frame, user_id='user_id',
                                                                             item_id='artist_name')

    # use turicreate to test various models
    ranking_model = turicreate.ranking_factorization_recommender.create(train_data, user_id='user_id',
                                                                        item_id='artist_name', verbose=False)
    print('ranking model built')
    item_model = turicreate.item_similarity_recommender.create(train_data, user_id='user_id', item_id='artist_name',
                                                               verbose=False, similarity_type='jaccard')
    print('item model built')
    popularity_model = turicreate.popularity_recommender.create(train_data, user_id='user_id', item_id='artist_name',
                                                                verbose=False)
    print('popularity model built')

    testing_analysis = turicreate.recommender.util.compare_models(test_data,
                                                                  models=[ranking_model, item_model, popularity_model],
                                                                  model_names=['ranking_model', 'item_model',

    print(item_model)
    print(testing_analysis)


def main():
    # handles loading all of the data and running the analysis
    # load listening data and get artists and their plays
    loaded_popular_artists = load_listening_data(listening_history_file)
    print(loaded_popular_artists.head())
    # load user data and merge with their listening history
    selected_users = load_user_data(full_user_profile_file, loaded_popular_artists)
    # free memory
    del loaded_popular_artists
    print(selected_users.head())
    # construct utility matrix for our data
    music_data = build_utility_matrix(selected_users, 'artist_name', 'user_id', 'user_play_count')
    print('built utility matrix')
    # build our nearest neighbor algorithm model
    matrix_file_path = '{}/{}'.format(working_directory, sparse_file_name)
    built_model = build_model(matrix_file_path, 'cosine')
    # make recommendations on random artists
    make_recommendations_random(matrix_file_path, music_data, built_model)
    # make recommendations on specific artists
    artists = ['radiohead', 'the beatles', 'drake', 'kanye west', 'the killers', 'beyonce', 'michael jackson', 'prince',
               'the script']
    make_recommendations_artists(music_data, built_model, artists)
    # test our recommender system
    test_recommender(music_data)


if __name__ == '__main__':
    main()
