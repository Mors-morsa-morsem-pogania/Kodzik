"""
    Plik zawiera:
        - wczytywanie nazw plików .wav do listy
        - tworzenie macierzy cech MFCC dla każdego pliku
        - funkcję zapisywania zmiennych do pliku .pkl
        - funkcję odczytywania plików .pkl do zmiennej

    [Uwaga1] Macierze MFCC zapisywane są w liście zawierającej również nazwę pliku z którego zostały stworzone.
    [Uwaga2] Pliki .pkl nie nadają się do odczytania przez człowieka, można nimi operować jedynie w zakresie
    biblioteki [pickle].

    Twórcy:
        Anna Lizińska, Paula Wajgelt
        AGH 2019
"""

from scipy.io import wavfile as wav
import numpy as np
from os import listdir
import pickle
import python_speech_features as psf

# train_path = input("Podaj ścieżkę dostępu do folderu z plikami treningowymi: ")
# lista_plikow = listdir(train_path)

#lista_plikow = listdir("E:\Studia\Semestr V\Technologia Mowy\ProjektI\Projekt_nr_1\\train\\")
lista_plikow = listdir("C:\\!STUDIA\\Technologia mowy\\PROJEKT 1 - Klasyfikacja cyfr\\train\\")


def tworzenie_listy_mfcc_nazwa(lista_plikow, filepath, czy_delta=False, czy_delta_delta=False):
    lista_mfcc_nazwa = []  # stworzenie pustej dużej listy

    for plik in range(0, len(lista_plikow)):
        nazwa = lista_plikow[plik]  # wydobycie nazwy pliku
        fs, dane_z_pliku = wav.read((filepath + nazwa))
        macierz_MFCC=psf.base.mfcc(dane_z_pliku, samplerate=fs,  winlen=0.025, winstep=0.01, numcep=13,
                                        nfilt=26, nfft=int(0.06*fs), lowfreq=0, highfreq=None, preemph=0.97,
                                        ceplifter=22, appendEnergy=True, winfunc=np.hamming)
        if czy_delta==True:
            macierz_MFCC_delta = psf.base.delta(macierz_MFCC, 2)
            macierz_MFCC = np.column_stack((macierz_MFCC, macierz_MFCC_delta))
            if czy_delta_delta==True:
                macierz_MFCC_delta_delta = psf.base.delta(macierz_MFCC_delta, 2)
                macierz_MFCC = np.column_stack((macierz_MFCC, macierz_MFCC_delta_delta))

        lista_mfcc_nazwa.append([macierz_MFCC, lista_plikow[plik]])  # dodanie mfcc i nazwy do listy
    return lista_mfcc_nazwa

# kiszenie do pliku:
def zapisywanie_kiszonki(filename, variable):
    if filename[-4:] != '.pkl':
        filename = filename + '.pkl'
    output = open(filename, 'wb')
    pickle.dump(variable, output)
    output.close()    #ważne!

def odczytywanie_kiszonki(filename):
    if filename[-4:] != '.pkl':
        filename = filename + '.pkl'
    pkl_file = open(filename, 'rb')
    variable = pickle.load(pkl_file)
    pkl_file.close()
    return variable


lista_mfcc_nazwa = tworzenie_listy_mfcc_nazwa(lista_plikow,
                                              filepath="C:\\!STUDIA\\Technologia mowy\\PROJEKT 1 - Klasyfikacja cyfr\\train\\",
                                              czy_delta=True, czy_delta_delta=False)
zapisywanie_kiszonki('lista_mfcc_nazwa.pkl', lista_mfcc_nazwa)

lista_plikow_eval = listdir("C:\\!STUDIA\\Technologia mowy\\PROJEKT 1 - Klasyfikacja cyfr\\eval\\")
lista_mfcc_nazwa_eval = tworzenie_listy_mfcc_nazwa(lista_plikow_eval,
                                                   filepath="C:\\!STUDIA\\Technologia mowy\\PROJEKT 1 - Klasyfikacja cyfr\\eval\\",
                                                   czy_delta=True, czy_delta_delta=False)
zapisywanie_kiszonki('lista_mfcc_nazwa_eval.pkl', lista_mfcc_nazwa_eval)