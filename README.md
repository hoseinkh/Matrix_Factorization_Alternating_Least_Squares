# Matrix Factorization - Alternating Least Squares

Here we implement Matrix Factorization using alternating least squares.

<br />

## Task:

The goal is to derive latent representation of the user and item feature vectors. The (predicted) ratings that a user gives to an item is the inner product of user's latent vector and the item's latent vector. We use alternating least squares for training.

We use the 20 million MovieLens data set available on [Kaggle](https://www.kaggle.com/grouplens/movielens-20m-dataset). Though, for practical implementation on a pc we shrink this dataset.

---

Matrix factorization plays a major role in the recommender systems. It:

- decreases the computations
- improves the performance as it increases the robustness of the system w.r.t. the noise.



In the following you can see how the matrix factorization works ([Ref](https://aws.amazon.com/blogs/machine-learning/build-a-movie-recommender-with-factorization-machines-on-amazon-sagemaker/))

<p float="left">
  <img src="/figs/MF_form.png" width="450" />
</p>



Now, we have two sets of vectors to determine: user-latent-features, and item-latent-features. We use alternating least squares to find them. The basic idea as follows:

#### Alternating Least Squares

1. Fix user's vector, use least squares to find item's vector.
2. Fix item's vector, use least squares to find user's vector.
3. Repeat 1-2 until converge.



---

### Codes & Results

The code consist of two parts. One is for the data preprocessing, and one implements and matrix factorization and gets the results.

<p float="left">
  <img src="/figs/MF_ALS_train_and_test_loss.png" width="450" />
</p>







------

### References

1. [Recommender Systems Handbook; Ricci, Rokach, Shapira](https://www.cse.iitk.ac.in/users/nsrivast/HCC/Recommender_systems_handbook.pdf)
2. [Statistical Methods for Recommender Systems; Agarwal, Chen](https://www.cambridge.org/core/books/statistical-methods-for-recommender-systems/0051A5BA0721C2C6385B2891D219ECD4)

