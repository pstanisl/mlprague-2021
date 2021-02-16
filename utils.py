import csv
import numpy as np

MOVIELENS_NUM_USERS = 943
MOVIELENS_NUM_MOVIES = 1682

def load_movielens_data(data_file):
  """Loads the movielens data and returns the ratings matrix."""
  ratings_matrix = np.zeros([MOVIELENS_NUM_USERS, MOVIELENS_NUM_MOVIES])
  with open(data_file, 'r') as infile:
    # The file is a csv with rows containing:
    # user id | item id | rating | timestamp
    reader = csv.reader(infile, delimiter='\t')
    for row in reader:
      user_id, item_id, rating, _ = row
      ratings_matrix[int(user_id) - 1, int(item_id) - 1] = float(rating)
  return ratings_matrix
