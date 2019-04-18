import tensorflow as tf

def gaussian_MLP_encoder(x, n_hidden, n_output, keep_prob):
...

  gaussian_params = tf.matmul(h1, wo) + bo
  mean = gaussian_params[:, :n_output]
  stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])
  
  return mean, stddev
 
def autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob):

  # encoding
  mu, sigma = gaussian_MLP_encoder(x_hat, n_hidden, dim_z, keep_prob)
  # sampling  by re-parameterization technique
  z = mu +sigma * tf.random_normal(tf.shape(mu, 0, 1, dtype=tf.float32))
  # decodding
  y = bernoulli_MLP_decoder(z, n_hidden, dim_img, keep_prob)
  
  # loss
  marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
  KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) -1, 1)
  
  ELBO = tf.reduce_mean(marginal_likelihood) - tf.reduce_mean(KL_divergence)
  loss = -ELBO
  
  
