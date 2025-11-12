# All Import Statements Defined Here
# Note: Do not add to this list.
# ----------------

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import scipy as sp
import random
import numpy as np
import re
from datasets import load_dataset
import matplotlib.pyplot as plt
import pprint
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from platform import python_version
import sys
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 8

assert int(python_version().split(".")[1]) >= 5, "Please upgrade your Python version following the instructions in \
    the README.md file found in the same directory as this notebook. Your Python version is " + python_version()

plt.rcParams['figure.figsize'] = [10, 5]

imdb_dataset = load_dataset("stanfordnlp/imdb", name="plain_text")


START_TOKEN = '<START>'
END_TOKEN = '<END>'
NUM_SAMPLES = 150

np.random.seed(0)
random.seed(0)
# ----------------
