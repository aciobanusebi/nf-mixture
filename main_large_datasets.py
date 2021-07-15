import sys
import numpy as np
import sklearn
from sklearn import metrics
import numpy as np
from sklearn.decomposition import PCA

def purity_score(y_true, y_pred): # from https://stackoverflow.com/a/51672699/7947996; in [0,1]; 0-bad,1-good
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

from sklearn.metrics.cluster import adjusted_rand_score # in [0,1]; 0-bad,1-good
from sklearn.metrics.cluster import normalized_mutual_info_score # in [0,1]; 0-bad,1-good

from coclust.evaluation.external import accuracy # in [0,1]; 0-bad,1-good; install it via "!pip install coclust" in Google Colab

def purity_score(y_true, y_pred): # from https://stackoverflow.com/a/51672699/7947996; in [0,1]; 0-bad,1-good
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

from sklearn.metrics.cluster import adjusted_rand_score # in [0,1]; 0-bad,1-good
from sklearn.metrics.cluster import normalized_mutual_info_score # in [0,1]; 0-bad,1-good

# !pip install coclust
# !pip install scikit-learn==0.22.2
from coclust.evaluation.external import accuracy # in [0,1]; 0-bad,1-good

def get_data_20news():
  import tensorflow as tf
  from sklearn.datasets import fetch_20newsgroups
  from sklearn.feature_extraction.text import TfidfVectorizer

  _20news = fetch_20newsgroups(subset="all")
  data = _20news.data
  target = _20news.target

  vectorizer = TfidfVectorizer(max_features=2000)
  data = vectorizer.fit_transform(data)
  data = data.toarray()

  return data, target


def get_data_mnist():
  import tensorflow as tf
  mnist = tf.keras.datasets.mnist
  (x_train, y_train),(x_test, y_test) = mnist.load_data()

  x_train = np.concatenate((x_train,x_test))
  y_train = np.concatenate((y_train,y_test))

  real_labels = y_train

  # # indices = np.isin(y_train,range(number_of_dist))
  # x_train = x_train[indices]
  # y_train = y_train[indices]

  samples = (x_train.reshape((x_train.shape[0],-1))/255.).astype(np.float32)
  
  return samples, real_labels

def get_data_cifar10():
  import tensorflow as tf
  mnist = tf.keras.datasets.cifar10
  (x_train, y_train),(x_test, y_test) = mnist.load_data()

  x_train = np.concatenate((x_train,x_test))
  y_train = np.concatenate((y_train,y_test))
  y_train = y_train.squeeze()

  real_labels = y_train

  # # indices = np.isin(y_train,range(number_of_dist))
  # x_train = x_train[indices]
  # y_train = y_train[indices]

  samples = (x_train.reshape((x_train.shape[0],-1))/255.).astype(np.float32)
  
  return samples, real_labels

def get_data_mnist5():
  import tensorflow as tf
  mnist = tf.keras.datasets.mnist
  (x_train, y_train),(x_test, y_test) = mnist.load_data()

  x_train = np.concatenate((x_train,x_test))
  y_train = np.concatenate((y_train,y_test))

  indices = y_train < 5
  x_train = x_train[indices]
  y_train = y_train[indices]

  real_labels = y_train
  
  samples = (x_train.reshape((x_train.shape[0],-1))/255.).astype(np.float32)
  
  return samples, real_labels

def get_data_fmnist():
  import tensorflow as tf
  mnist = tf.keras.datasets.fashion_mnist
  (x_train, y_train),(x_test, y_test) = mnist.load_data()

  x_train = np.concatenate((x_train,x_test))
  y_train = np.concatenate((y_train,y_test))

  real_labels = y_train

  # # indices = np.isin(y_train,range(number_of_dist))
  # x_train = x_train[indices]
  # y_train = y_train[indices]

  samples = (x_train.reshape((x_train.shape[0],-1))/255.).astype(np.float32)
  
  return samples, real_labels

def get_data_usps():
  import h5py
  path = "./usps.h5"
  with h5py.File(path, 'r') as hf:
    train = hf.get('train')
    X_tr = train.get('data')[:]
    y_tr = train.get('target')[:]
    test = hf.get('test')
    X_te = test.get('data')[:]
    y_te = test.get('target')[:]

  samples = np.concatenate((X_tr,X_te))
  real_labels = np.concatenate((y_tr,y_te))
  return samples, real_labels

def nll(distribution,samples):
          return -tf.reduce_mean(distribution.log_prob(samples))

def get_loss_and_grads(distribution,samples):
  with tf.GradientTape() as tape:
    tape.watch(distribution.trainable_variables)
    # print(distribution.trainable_variables)
    loss = nll(distribution,samples)
    grads = tape.gradient(loss, distribution.trainable_variables)
  return loss, grads


import sys
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from tqdm import tqdm
import statistics
import numpy as np
import matplotlib.pyplot as plt

tfk = tf.keras
tfkl = tf.keras.layers
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
tfd = tfp.distributions


original_data_name = sys.argv[1]
dim_pca = sys.argv[2]
if dim_pca == "None":
  dim_pca = None
else:
  dim_pca = int(dim_pca)
it = int(sys.argv[3])


print(it)

if original_data_name == "mnist":
  samples, real_labels = get_data_mnist()
elif original_data_name == "mnist5":
  samples, real_labels = get_data_mnist5()
elif original_data_name == "cifar10":
  samples, real_labels = get_data_cifar10()
elif original_data_name == "fmnist":
  samples, real_labels = get_data_fmnist()
elif original_data_name == "20news":
  samples, real_labels = get_data_20news()
elif original_data_name == "usps":
  samples, real_labels = get_data_usps()

k = len(np.unique(real_labels))

if dim_pca is not None:
  X = samples
  pca = PCA(n_components=dim_pca)
  samples = pca.fit_transform(X)

BATCH_SIZE = 1000 # for cifar10 None: 1000; for fmnist None: 3500; for mnist None: 4000; the rest: 5000
SHUFFLE_BUFFER_SIZE = 1024
D = samples.shape[1]

dataset_aux = tf.data.Dataset.from_tensor_slices(samples)
dataset = dataset_aux.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
dataset_not_shuffled = dataset_aux.batch(BATCH_SIZE)

epochs = 10

optimizer = tf.keras.optimizers.Adam()

distribution = tfd.Mixture(
  cat=tfd.Categorical(
      logits=tf.Variable([1.0 for i in range(k)],dtype=tf.float32)),
  components=[tfd.TransformedDistribution(
  distribution=tfd.MultivariateNormalDiag(
        loc=np.random.rand(D).astype(np.float32),#kmeans.cluster_centers_[i],
        scale_diag=[0.1 for _ in range(D)]),#scale_diag=np.std(samples,axis=0)),
  bijector=tfb.MaskedAutoregressiveFlow(
      tfb.AutoregressiveNetwork(params=2, hidden_units=[1024],activation="tanh")
  )) for i in range(k)])
  # bijector=tfb.RealNVP(
  #       num_masked=1,
  #       shift_and_log_scale_fn=tfb.real_nvp_default_template(
  #           hidden_layers=[512,512],activation="relu"))) for i in range(k)])



losses = []
for _ in (range(epochs)):
#   print(_)
  mean_loss = 0
  count = 0
  for batch in dataset:
    loss, grads = get_loss_and_grads(distribution,batch)
    del batch
    optimizer.apply_gradients(zip(grads, distribution.trainable_variables))
    mean_loss += loss
    count+=1
    if np.isnan(loss):
      break
    del loss
    del grads
  mean_loss/=count
  # print(mean_loss.numpy())
  losses.append(mean_loss)
# plt.figure()
# plt.plot(losses)
# plt.show()
# it_losses.append(losses[-1])
print(losses[-1].numpy())
dist_cat_log_probs = [distribution.cat.log_prob(i) for i in range(k)]
clustering=[]
counter=0
for batch in dataset_not_shuffled:
  counter+=1
  # print(counter)
  dist_components_weigthed_log_probs = np.zeros((k,len(batch)))
  for i in range(k):
    dist_components_weigthed_log_probs[i,:] = dist_cat_log_probs[i] + distribution.components[i].log_prob(batch)
  clustering += list(np.argmax(dist_components_weigthed_log_probs, axis=0))


print(purity_score(real_labels,clustering))
print(adjusted_rand_score(real_labels,clustering))
print(normalized_mutual_info_score(real_labels,clustering))
print(accuracy(real_labels,clustering))

matrix = sklearn.metrics.cluster.contingency_matrix(real_labels, clustering)
print(matrix)
print(matrix/matrix.sum(axis=1, keepdims=True))

# print(np.argmin(it_losses))