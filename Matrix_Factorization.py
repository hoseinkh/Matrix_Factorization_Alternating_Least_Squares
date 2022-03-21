# =========================================================
# For more info, see https://hoseinkh.github.io/projects/
# =========================================================
import pickle
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
## ********************************************************
## Parameters
K = 15 # latent dimensionality
# train the parameters
epochs = 20
reg = 25 # regularization penalty
## ********************************************************
# load in the data
with open('./Data/user_to_movie.json', 'rb') as f:
  user_to_movie = pickle.load(f)
##
with open('./Data/movie_to_user.json', 'rb') as f:
  movie_to_user = pickle.load(f)
##
with open('./Data/user_and_movie_to_rating.json', 'rb') as f:
  user_and_movie_to_rating = pickle.load(f)
##
with open('./Data/user_and_movie_to_rating___test_data.json', 'rb') as f:
  user_and_movie_to_rating___test_data = pickle.load(f)
## ********************************************************
N_max_user_id_in_train = np.max(list(user_to_movie.keys())) + 1
# the test set may contain movies the train set doesn't have data on
m1_max_movie_id_in_train = np.max(list(movie_to_user.keys()))
m2_max_movie_id_in_test = np.max([m for (u, m), r in user_and_movie_to_rating___test_data.items()])
M_max_movie_id_in_tain_and_test = max(m1_max_movie_id_in_train, m2_max_movie_id_in_test) + 1
print("num_users:", N_max_user_id_in_train, "num_movies:", M_max_movie_id_in_tain_and_test)
## ********************************************************
# prediction[i,j] = W[i].dot(U[j]) + b[i] + c.T[j] + mu
def get_loss(d):
  # d = dictionary: (user_id, movie_id) -> rating
  num_rating_in_d = float(len(d))
  SSE = 0
  for id_tuple, actual_rating in d.items():
    u, m = id_tuple[0], id_tuple[1]
    pred_rating = W[u].dot(U[m]) + b[u] + c[m] + mu
    SSE += (pred_rating - actual_rating)**2
  return SSE / num_rating_in_d
## ********************************************************
## ****************************
## Training
#
# initialize variables
W = np.random.randn(N_max_user_id_in_train, K) # weights for the users
b = np.zeros(N_max_user_id_in_train)           # bias terms for the users
U = np.random.randn(M_max_movie_id_in_tain_and_test, K) # weights for the movies
c = np.zeros(M_max_movie_id_in_tain_and_test)           # bias terms for the movies
## mu is the average of the user-item matrix --- we can instantly calculate it as :
mu = np.mean(list(user_and_movie_to_rating.values()))   # the global bias term of the rating matrix
#
## Begining of the calculations for the updates
train_losses = []
test_losses = []
for epoch in tqdm(range(epochs)):
  ## ****************************
  ## perform updates using Alternative Least Square Algorithm!
  #
  ## update W and b (weight and bias for the users)
  for i in range(N_max_user_id_in_train):
    # for W
    matrix = np.eye(K) * reg
    vector = np.zeros(K)
    #
    # for b
    bi = 0
    for j in user_to_movie[i]:
      r = user_and_movie_to_rating[(i,j)]
      matrix += np.outer(U[j], U[j])
      vector += (r - b[i] - c[j] - mu)*U[j]
      bi += (r - W[i].dot(U[j]) - c[j] - mu)
    #
    # apply the updates
    W[i] = np.linalg.solve(matrix, vector)
    b[i] = bi / (len(user_to_movie[i]) + reg)
    #
  #
  # update U and c  (weight and bias for the movies)
  for j in range(M_max_movie_id_in_tain_and_test):
    # for U
    matrix = np.eye(K) * reg
    vector = np.zeros(K)
    #
    # for c
    cj = 0
    try:
      for i in movie_to_user[j]:
        r = user_and_movie_to_rating[(i,j)]
        matrix += np.outer(W[i], W[i])
        vector += (r - b[i] - c[j] - mu)*W[i]
        cj += (r - W[i].dot(U[j]) - b[i] - mu)
      #
      # Apply the updates
      U[j] = np.linalg.solve(matrix, vector)
      c[j] = cj / (len(movie_to_user[j]) + reg)
      #
    except KeyError:
      # possible not to have any ratings for a movie
      pass
  #
  #
  ## store train loss
  train_losses.append(get_loss(user_and_movie_to_rating))
  #
  ## store test loss
  test_losses.append(get_loss(user_and_movie_to_rating___test_data))
  #
  print("train loss:", train_losses[-1])
  print("test loss:", test_losses[-1])
## ****************************
print("train losses:", train_losses)
print("test losses:", test_losses)
#
# plot losses
plt.plot(train_losses, label="train loss")
plt.plot(test_losses, label="test loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.savefig("./figs/MF_train_and_test_loss.png")
plt.show()
#
