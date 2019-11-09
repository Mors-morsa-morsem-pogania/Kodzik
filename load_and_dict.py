import wave
from scipy.io import wavfile as wav
import numpy as np
from os import listdir
import scipy
import matplotlib.pyplot as plt
import sklearn
import sklearn.mixture
import pickle

# odczyt z pliku listy mfcc_nazwa
nazwa_pliku = 'lista_mfcc_nazwa.pkl'
pkl_file = open(nazwa_pliku, 'rb')

data = pickle.load(pkl_file)
pkl_file.close()

