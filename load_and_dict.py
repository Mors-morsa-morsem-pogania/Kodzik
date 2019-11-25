"""
    Plik zawiera:
        - tworzenie słownika na bazie mówców ( [mówca] -> [macierz MFCC, nazwa pliku] )
        - tworzenie listy macierzy MFCC wszystkich cyfr (potrzebne do liczenia prawdopodobieństwa)
        - funkcję tworzenia modelu GMM na bazie zadanej macierzy MFCC
        - funkcję liczenia prawdopodobieństwa dla zadanej cyfry na bazie log-likelihood modelu

    Twórcy:
        Anna Lizińska, Paula Wajgelt
        AGH 2019
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn
import sklearn.mixture
import sklearn.model_selection
import pickle
from collections import defaultdict

from wczytywnie_plikow import tworzenie_listy_mfcc_nazwa, lista_plikow, zapisywanie_kiszonki, odczytywanie_kiszonki


def stworz_slownik(pliki, mfcc_nazwa):
    slownik = defaultdict(list)  # stworzenie pustego słownika
    for i in range(0, 21):
        if i == 0:
            # wybranie 10 cyfr dla danego mówcy i przypisanie ich do klucza (mówcy w słowniku)
            slownik[str(pliki[i * 10 + 1])[0:3]] = mfcc_nazwa[0:10]
        else:
            slownik[str(pliki[i * 10 + 1])[0:3]] = mfcc_nazwa[(10 * i):(10 * i + 10)]
    return slownik


def wszystkie_cyfry(nr_cyfry,lista_mfcc_nazwa):
    cyfry=[]

    for i in range(0, int(len(lista_mfcc_nazwa)/10)):
        cyfry.extend(lista_mfcc_nazwa[i * 10+nr_cyfry][0])
    cyfry = np.array(cyfry)
    return cyfry


def GaussianMix(macierz, n_comp=8, n_iter=100, n_init=1, random_state=None):
    gmm=sklearn.mixture.GaussianMixture(n_components=n_comp, covariance_type='diag', tol=0.001, reg_covar=1e-06,
                                            max_iter=n_iter, n_init=n_init, init_params='kmeans', weights_init=None,
                                            means_init=None, precisions_init=None, random_state=random_state,
                                            warm_start=False, verbose=0, verbose_interval=10)
    gmm.fit(macierz)
    return gmm


def prawdopodobienstwo_cyfry(gmmx, macierz_mfcc_cyfry, lista_mfcc_nazwa):
    lista_cyfr = []
    for cyfra in range(0, 10):
        lista_cyfr.append(wszystkie_cyfry(cyfra, lista_mfcc_nazwa))
    wszystkie_GMM = []
    for gaussian in range(0, 10):
        wszystkie_GMM.append(GaussianMix(lista_cyfr[gaussian], n_comp=10,
                                         n_iter=150, n_init=5, random_state=None))

    p = np.exp(gmmx.score(macierz_mfcc_cyfry)) * 0.1 / (np.exp(wszystkie_GMM[0].score(macierz_mfcc_cyfry)) * 0.1
                                                        + np.exp(wszystkie_GMM[1].score(macierz_mfcc_cyfry)) * 0.1
                                                        + np.exp(wszystkie_GMM[2].score(macierz_mfcc_cyfry)) * 0.1
                                                        + np.exp(wszystkie_GMM[3].score(macierz_mfcc_cyfry)) * 0.1
                                                        + np.exp(wszystkie_GMM[4].score(macierz_mfcc_cyfry)) * 0.1
                                                        + np.exp(wszystkie_GMM[5].score(macierz_mfcc_cyfry)) * 0.1
                                                        + np.exp(wszystkie_GMM[6].score(macierz_mfcc_cyfry)) * 0.1
                                                        + np.exp(wszystkie_GMM[7].score(macierz_mfcc_cyfry)) * 0.1
                                                        + np.exp(wszystkie_GMM[8].score(macierz_mfcc_cyfry)) * 0.1
                                                        + np.exp(wszystkie_GMM[9].score(macierz_mfcc_cyfry)) * 0.1)
    return p

# p=[]
# for i in range(0,len(wszystkie_GMM)):
#     p.append(prawdopodobienstwo_cyfry(wszystkie_GMM[i], wszystkie_cyfry_0))
#     print("log(p(MFCC_0 | GMM_"+str(i)+" = ", p[i])
#
# print(sum(p)) #sumują sie do 1 - sztos

# zmienna pomocnicza
a=1