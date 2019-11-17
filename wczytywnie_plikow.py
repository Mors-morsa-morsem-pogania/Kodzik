"""kluczem będzie mówca, w liście dla jednego mówcy mają być wypisane wszystkie MFCC dla niego i lista nazw plików
Wszystko będzie w jednym słowniku.
dict [speakerID] = ( tablicaMFCC, lista_filename)"""

from scipy.io import wavfile as wav
import numpy as np
from os import listdir
import pickle
import python_speech_features as psf

#lista_plikow = listdir("E:\Studia\Semestr V\Technologia Mowy\ProjektI\Projekt_nr_1\\train\\")
lista_plikow = listdir("C:\\!STUDIA\\Technologia mowy\\PROJEKT 1 - Klasyfikacja cyfr\\train\\")

lista_mfcc_nazwa = []  # stworzenie pustej dużej listy

for i in range(0, len(lista_plikow)):
    nazwa = lista_plikow[i]  # wydobycie nazwy pliku
    fs, dane_z_pliku = wav.read(("C:\\!STUDIA\\Technologia mowy\\PROJEKT 1 - Klasyfikacja cyfr\\train\\" + nazwa))
    macierz_MFCC=psf.base.mfcc(dane_z_pliku, samplerate=fs,  winlen=0.025, winstep=0.01, numcep=13,
                                    nfilt=10, nfft=int(0.06*fs), lowfreq=0, highfreq=None, preemph=0.1,
                                    ceplifter=0, appendEnergy=True, winfunc=np.hamming)
    lista_mfcc_nazwa.append([macierz_MFCC, lista_plikow[i]])  # dodanie mfcc i nazwy do listy


# kiszenie do pliku:
output = open('lista_mfcc_nazwa.pkl', 'wb')
pickle.dump(lista_mfcc_nazwa, output)
output.close()    #ważne!
