import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def read_dataset():
	df pd.read_csv("/home/ananth/naive.data")
	x = df[df.columns[0:4]].values
	y = df[df.columns[4]]

	encoder = LabelEncoder()
	encoder.fit(y)
	y = encoder.transform(y)
	Y = one_hot_encode(y)
	print(X.shape)
	return (X,Y)
