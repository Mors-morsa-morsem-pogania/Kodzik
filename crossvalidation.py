"""
    Twórcy:
        Anna Lizińska, Paula Wajgelt
        AGH 2019
"""
from load_and_dict import wszystkie_cyfry, GaussianMix
import numpy as np
from wczytywnie_plikow import lista_plikow
import sklearn


def accuracy_percent(slownik, n_splits, n_comp, n_iter, n_init, random_state):
    """
    Funkcja obliczająca średni procent dokładności z jaką model rozpoznaje poszczególne cyfry.

    :param slownik: Słownik zawierający macierze MFCC na bazie mówców
    :param n_splits: Parametr funkcji KFold o tej samej nazwie
            Pozostałe parametry wejściowe dotyczą GaussianMix zawartego w funkcji.
    :return: Średnia arytmetyczna z Accuracy [%]

    """
    xvalid=sklearn.model_selection.KFold(n_splits=n_splits)
    xvalid.get_n_splits(slownik)
    klucze=[*slownik]

    acc=[]
    for train_ids, test_ids in xvalid.split(slownik):
        X_train = {} #pusty słownik na dane treningowe
        X_test = {} #pusty słownik na dane testowe
        lista_mfcc_nazwa_train=[] #pusta lista na dane treningowe
        lista_mfcc_nazwa_test=[] #pusta lista na dane testowe

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
            lista_gaussianow_train.append(GaussianMix(lista_cyfr_train[gaussian], n_comp=n_comp,
                                                      n_iter=n_iter, n_init=n_init, random_state=random_state))

        p = []
        y_pred = []
        for trening in range(0,10):
            for test in range(0,10):
                p.append(lista_gaussianow_train[trening].score(lista_cyfr_test[test]))
            y_pred.append(p.index(max(p)))
            p = []

        y_true = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        acc.append(sklearn.metrics.accuracy_score(y_true,y_pred, normalize=True))
    return np.mean(acc)

def co_to_jest_za_cyfra_wg_naszego_modelu(lista_mfcc_nazwa_eval, lista_mfcc_nazwa):
    lista_cyfr = []
    for cyfra in range(0, 10):
        lista_cyfr.append(wszystkie_cyfry(cyfra, lista_mfcc_nazwa))

    wszystkie_GMM = []
    for gaussian in range(0, 10):
        wszystkie_GMM.append(GaussianMix(lista_cyfr[gaussian], n_comp=10,
                                        n_iter=150, n_init=5, random_state=None))
    p=[]
    y_pred=[]
    p_score=[]
    for plik in range(0,len(lista_mfcc_nazwa_eval)):

        for i in range(0, 10):
            p.append(wszystkie_GMM[i].score(lista_mfcc_nazwa_eval[plik][0]))
        y_pred.append(p.index(max(p)))
        p_score.append(max(p))
        p = []

    return y_pred, p_score