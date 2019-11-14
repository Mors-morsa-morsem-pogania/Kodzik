import wave
from scipy.io import wavfile as wav
import numpy as np
from os import listdir
import scipy
import matplotlib.pyplot as plt
import sklearn
import sklearn.mixture
import pickle
from collections import defaultdict

# odczyt z pliku listy mfcc_nazwa
nazwa_pliku = 'lista_mfcc_nazwa.pkl'
pkl_file = open(nazwa_pliku, 'rb')

lista_mfcc_nazwa = pickle.load(pkl_file)
pkl_file.close()

# tworzenie słownika

slownik = defaultdict(list)  # stworzenie pustego słownika

lista_plikow=[] #stworzenie pustej listy i wydobycie nazw ze środka
for i in range(0,len(lista_mfcc_nazwa)-1):
    lista_plikow.append(lista_mfcc_nazwa[i][1]) 

for i in range(0, 22):
    if i == 0:
        slownik[str(lista_plikow[i * 10 + 1])[0:3]] = lista_mfcc_nazwa[
                                                      0:10]  # wybranie 10 cyfr dla danego mówcy i przypisanie ich do klucza (mówcy w słowniku)
    else:
        slownik[str(lista_plikow[i * 10 + 1])[0:3]] = lista_mfcc_nazwa[10 * i + 1:10 * (i + 1)]

print(len(slownik))  # dlugość 22 - zgadza sie
print(slownik["AO1"][1][1])

# Przydatne:
# # Odwołanie sie do wiersza w słowniku: np. slownik["AO1"]
# print(slownik["AO1"])
# # Odwołanie sie do wiersza w liście która jest w słowniku: np. slownik["AO1"][0]
# print(slownik["AO1"][0])
# # Odwołanie sie do macierzy MFCC w liście w słowniku: np. slownik["AO1"][0][0]
# print(slownik["AO1"][0][0])
# # Odwołanie sie do nazwy całego pliku w liście w słowniku: np. slownik["AO1"][0][1]
# print(slownik["AO1"][0][1])

def wszystkie_cyfry(nr_cyfry,lista_mfcc_nazwa):
    cyfry=[]
    for i in range(0, 22):
        cyfry.extend(lista_mfcc_nazwa[i * 10+nr_cyfry][0])
    cyfry = np.array(cyfry)
    return cyfry

wszystkie_cyfry_0=wszystkie_cyfry(0,lista_mfcc_nazwa)
wszystkie_cyfry_1=wszystkie_cyfry(1,lista_mfcc_nazwa)
wszystkie_cyfry_2=wszystkie_cyfry(2,lista_mfcc_nazwa)
wszystkie_cyfry_3=wszystkie_cyfry(3,lista_mfcc_nazwa)
wszystkie_cyfry_4=wszystkie_cyfry(4,lista_mfcc_nazwa)
wszystkie_cyfry_5=wszystkie_cyfry(5,lista_mfcc_nazwa)
wszystkie_cyfry_6=wszystkie_cyfry(6,lista_mfcc_nazwa)
wszystkie_cyfry_7=wszystkie_cyfry(7,lista_mfcc_nazwa)
wszystkie_cyfry_8=wszystkie_cyfry(8,lista_mfcc_nazwa)
wszystkie_cyfry_9=wszystkie_cyfry(9,lista_mfcc_nazwa)

def GaussianMix(macierz,n,n_iter):
    gmm=sklearn.mixture.GaussianMixture(n_components=n, covariance_type='diag', tol=0.001, reg_covar=1e-06,
                                            max_iter=n_iter, n_init=1, init_params='kmeans', weights_init=None,
                                            means_init=None, precisions_init=None, random_state=None,
                                            warm_start=False, verbose=0, verbose_interval=10)
    gmm.fit(macierz)
    return gmm

GMM_0=GaussianMix(wszystkie_cyfry_0,8,100)
GMM_1=GaussianMix(wszystkie_cyfry_1,8,100)
GMM_2=GaussianMix(wszystkie_cyfry_2,8,100)
GMM_3=GaussianMix(wszystkie_cyfry_3,8,100)
GMM_4=GaussianMix(wszystkie_cyfry_4,8,100)
GMM_5=GaussianMix(wszystkie_cyfry_5,8,100)
GMM_6=GaussianMix(wszystkie_cyfry_6,8,100)
GMM_7=GaussianMix(wszystkie_cyfry_7,8,100)
GMM_8=GaussianMix(wszystkie_cyfry_8,8,100)
GMM_9=GaussianMix(wszystkie_cyfry_9,8,100)

wszystkie_GMM=[GMM_0,GMM_1,GMM_2,GMM_3,GMM_4,GMM_5,GMM_6,GMM_7,GMM_8,GMM_9]

def prawdopodobienstwo_cyfry(gmmx,mfcc_cyfry):
    p = np.exp(gmmx.score(mfcc_cyfry)) * 0.1 / (np.exp(GMM_0.score(mfcc_cyfry)) * 0.1 + np.exp(GMM_1.score(mfcc_cyfry)) * 0.1
                                             + np.exp(GMM_2.score(mfcc_cyfry)) * 0.1 + np.exp(GMM_3.score(mfcc_cyfry)) * 0.1
                                             + np.exp(GMM_4.score(mfcc_cyfry)) * 0.1 + np.exp(GMM_5.score(mfcc_cyfry)) * 0.1
                                             + np.exp(GMM_6.score(mfcc_cyfry)) * 0.1 + np.exp(GMM_7.score(mfcc_cyfry)) * 0.1
                                             + np.exp(GMM_8.score(mfcc_cyfry)) * 0.1 + np.exp(GMM_9.score(mfcc_cyfry)) * 0.1)
    return p

p=[]
for i in range(0,len(wszystkie_GMM)):
    p.append(prawdopodobienstwo_cyfry(wszystkie_GMM[i], wszystkie_cyfry_0))
    print("log(p(MFCC_0 | GMM_"+str(i)+" = ", p[i])

print(sum(p)) #sumują sie do 1 - sztos

xvalid=sklearn.model_selection.KFold(n_splits=5)
xvalid.get_n_splits(slownik)
# klucze=list(slownik.keys())
klucze=[*slownik]
print(klucze)
print(xvalid)

for train_ids, test_ids in xvalid.split(slownik):
    X_train = {}
    X_test = {}
    print("Train: ", train_ids, "Test: ", test_ids)
    for train in train_ids:
        X_train[klucze[train]]=slownik[klucze[train]]
    for test in test_ids:
        X_test[klucze[test]] = slownik[klucze[test]]
    print(len(X_train),len(X_test))
    #przejść po wszstkich trainach i testach na raz
    #nowy model będzie stworzony tylko dla mfcc dla mówców z pociągu
    #stworzenie nowej turbo macierzy mfcc dla pociągu i oddzielnie dla testowych
    #dwa podsłowniki
