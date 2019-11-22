from wczytywnie_plikow import lista_plikow
from wczytywnie_plikow import tworzenie_listy_mfcc_nazwa, odczytywanie_kiszonki, zapisywanie_kiszonki
from load_and_dict import stworz_slownik, prawdopodobienstwo_cyfry
from crossvalidation import accuracy_percent

lista_mfcc_nazwa = tworzenie_listy_mfcc_nazwa(lista_plikow, czy_delta=False, czy_delta_delta=False)
slownik = stworz_slownik(lista_plikow, lista_mfcc_nazwa)
acc = accuracy_percent(slownik=slownik, n_splits=5, n_comp=8, n_iter=100, n_init=1, random_state=2)
print('1. '+str(acc))

lista_mfcc_nazwa = tworzenie_listy_mfcc_nazwa(lista_plikow, czy_delta=False, czy_delta_delta=False)
slownik = stworz_slownik(lista_plikow, lista_mfcc_nazwa)
acc = accuracy_percent(slownik=slownik, n_splits=5, n_comp=10, n_iter=100, n_init=1, random_state=2)
print('2. '+str(acc))

lista_mfcc_nazwa = tworzenie_listy_mfcc_nazwa(lista_plikow, czy_delta=False, czy_delta_delta=False)
slownik = stworz_slownik(lista_plikow, lista_mfcc_nazwa)
acc = accuracy_percent(slownik=slownik, n_splits=5, n_comp=10, n_iter=150, n_init=1, random_state=2)
print('3. '+str(acc))
# wniosek - zmiana liczby iteracji nie ma większego wpływu na dokładność
# #(prawdopodobnie optimum jest osiągane w mniejszej liczbie iteracji)

lista_mfcc_nazwa = tworzenie_listy_mfcc_nazwa(lista_plikow, czy_delta=False, czy_delta_delta=False)
slownik = stworz_slownik(lista_plikow, lista_mfcc_nazwa)
acc = accuracy_percent(slownik=slownik, n_splits=5, n_comp=10, n_iter=100, n_init=5, random_state=2)
print('4. '+str(acc))

lista_mfcc_nazwa = tworzenie_listy_mfcc_nazwa(lista_plikow, czy_delta=False, czy_delta_delta=False)
slownik = stworz_slownik(lista_plikow, lista_mfcc_nazwa)
acc = accuracy_percent(slownik=slownik, n_splits=10, n_comp=10, n_iter=100, n_init=5, random_state=2)
print('5. '+str(acc))

lista_mfcc_nazwa = tworzenie_listy_mfcc_nazwa(lista_plikow, czy_delta=True, czy_delta_delta=False)
slownik = stworz_slownik(lista_plikow, lista_mfcc_nazwa)
acc = accuracy_percent(slownik=slownik, n_splits=5, n_comp=10, n_iter=100, n_init=5, random_state=2)
print('6. '+str(acc))




