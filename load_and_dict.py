import wave
import numpy as np
from os import listdir
import scipy
import matplotlib.pyplot as plt
import sklearn
import sklearn.mixture
import sklearn.model_selection
from scipy.io import wavfile as wav
from os import listdir
import pickle
from collections import defaultdict
import python_speech_features as psf

nazwa_pliku = 'lista_mfcc_nazwa.pkl'
pkl_file = open(nazwa_pliku, 'rb')

lista_mfcc_nazwa = pickle.load(pkl_file)
pkl_file.close()

slownik = defaultdict(list)  # stworzenie pustego słownika
lista_plikow=[]
for i in range(0,len(lista_mfcc_nazwa)-1):
    lista_plikow.append(lista_mfcc_nazwa[i][1]) # nie wiem jak to wydobyć ze środka :/

# print(lista_plikow)
for i in range(0, 21):
    if i == 0:
        slownik[str(lista_plikow[i * 10 + 1])[0:3]] = lista_mfcc_nazwa[0:10]  # wybranie 10 cyfr dla danego mówcy i przypisanie ich do klucza (mówcy w słowniku)
    else:
        slownik[str(lista_plikow[i * 10 + 1])[0:3]] = lista_mfcc_nazwa[(10 * i):(10 * i + 10)]

print(len(slownik))  # dlugość 22 - zgadza sie
print(slownik["AO1"][1][1])

def wszystkie_cyfry(nr_cyfry,lista_mfcc_nazwa):
    cyfry=[]

    for i in range(0, int(len(lista_mfcc_nazwa)/10)):
        cyfry.extend(lista_mfcc_nazwa[i * 10+nr_cyfry][0])
    cyfry = np.array(cyfry)
    return cyfry

lista_cyfr=[]
for cyfra in range(0,10):
    lista_cyfr.append(wszystkie_cyfry(cyfra,lista_mfcc_nazwa))

def GaussianMix(macierz,n,n_iter):
    gmm=sklearn.mixture.GaussianMixture(n_components=n, covariance_type='diag', tol=0.001, reg_covar=1e-06,
                                            max_iter=n_iter, n_init=5, init_params='kmeans', weights_init=None,
                                            means_init=None, precisions_init=None, random_state=None,
                                            warm_start=False, verbose=0, verbose_interval=10)
    gmm.fit(macierz)
    return gmm

wszystkie_GMM=[]
for gaussian in range(0,10):
    wszystkie_GMM.append(GaussianMix(lista_cyfr[gaussian],8,100))

def prawdopodobienstwo_cyfry(gmmx,mfcc_cyfry):
    p = np.exp(gmmx.score(mfcc_cyfry)) * 0.1 / (np.exp(wszystkie_GMM[0].score(mfcc_cyfry)) * 0.1 + np.exp(wszystkie_GMM[1].score(mfcc_cyfry)) * 0.1
                                             + np.exp(wszystkie_GMM[2].score(mfcc_cyfry)) * 0.1 + np.exp(wszystkie_GMM[3].score(mfcc_cyfry)) * 0.1
                                             + np.exp(wszystkie_GMM[4].score(mfcc_cyfry)) * 0.1 + np.exp(wszystkie_GMM[5].score(mfcc_cyfry)) * 0.1
                                             + np.exp(wszystkie_GMM[6].score(mfcc_cyfry)) * 0.1 + np.exp(wszystkie_GMM[7].score(mfcc_cyfry)) * 0.1
                                             + np.exp(wszystkie_GMM[8].score(mfcc_cyfry)) * 0.1 + np.exp(wszystkie_GMM[9].score(mfcc_cyfry)) * 0.1)
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

acc=[]
for train_ids, test_ids in xvalid.split(slownik):
    X_train = {} #pusty słownik na dane treningowe
    X_test = {} #pusty słownik na dane testowe
    lista_mfcc_nazwa_train=[] #pusta lista na dane treningowe
    lista_mfcc_nazwa_test=[] #pusta lista na dane testowe
    #print("Train: ", train_ids, "Test: ", test_ids)
    for train in train_ids:
        X_train[klucze[train]]=slownik[klucze[train]] #slownik z danymi treningowymi
        for i in range(0,10):
            lista_mfcc_nazwa_train.append([X_train[klucze[train]][i][0],X_train[klucze[train]][i][1]]) #lista danych treningowych mfcc + nazwa pliku
    for test in test_ids:
        X_test[klucze[test]] = slownik[klucze[test]] #slownik z danymi testowymi
        for i in range(0, 10):
            lista_mfcc_nazwa_test.append([X_test[klucze[test]][i][0], X_test[klucze[test]][i][1]])  #lista danych testowych mfcc + nazwa pliku

    lista_cyfr_train=[]
    for c in range(0,10):
        lista_cyfr_train.append(wszystkie_cyfry(c,lista_mfcc_nazwa_train)) #duże macierze mfcc dla każdej z cyfr z danych treninowych

    lista_cyfr_test = []
    for c in range(0, 10):
        lista_cyfr_test.append(wszystkie_cyfry(c, lista_mfcc_nazwa_test)) #duże macierze mfcc dla każdej z cyfr z danych testowych

    lista_gaussianow_train=[] #lista modeli dla cyfr od 0 do 9
    for gaussian in range(0,10):
        lista_gaussianow_train.append(GaussianMix(lista_cyfr_train[gaussian],10,150))

    p=[]
    y_pred=[]
    for trening in range(0,10):
        for test in range(0,10):
            p.append(lista_gaussianow_train[trening].score(lista_cyfr_test[test]))
        y_pred.append(p.index(max(p)))
        p=[]

    y_true = [0,1,2,3,4,5,6,7,8,9]
    acc.append(sklearn.metrics.accuracy_score(y_true,y_pred, normalize=True))
    #potrzebowałam tego do optymalizacji, i za nic w świecie nie umiem podniesć skuteczności sieci powyzej 82% #edit podbiłam
print(np.mean(acc))


a=1
