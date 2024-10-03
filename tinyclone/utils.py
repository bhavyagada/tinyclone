import numpy as np

def layer_init_uniform(m, h):
  ret = np.random.uniform(-1., 1., size=(m, h))/np.sqrt(m*h)
  return ret.astype(np.float32)

def fetch_mnist():
  def fetch(url):
    import requests, gzip, os, hashlib, numpy
    download_dir = os.path.join(os.getcwd(), "downloads")
    os.makedirs(download_dir, exist_ok=True)
    fp = os.path.join(download_dir, hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
      with open(fp, "rb") as f:
        dat = f.read()
    else:
      dat = requests.get(url).content
      with open(fp, "wb") as f:
        f.write(dat)
    return numpy.frombuffer(gzip.decompress(dat), dtype=numpy.uint8).copy()
  X_train = fetch("https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
  Y_train = fetch("https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz")[8:]
  X_test = fetch("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
  Y_test = fetch("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz")[8:]

  return X_train, Y_train, X_test, Y_test
