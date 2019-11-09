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

lista_mfcc_nazwa = pickle.load(pkl_file)
pkl_file.close()

# tworzenie słownika

slownik = {}  # stworzenie pustego słownika

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





