from wczytywnie_plikow import lista_plikow
from wczytywnie_plikow import tworzenie_listy_mfcc_nazwa, odczytywanie_kiszonki, zapisywanie_kiszonki
from load_and_dict import stworz_slownik, wszystkie_cyfry, GaussianMix, prawdopodobienstwo_cyfry
from crossvalidation import accuracy_percent, co_to_jest_za_cyfra_wg_naszego_modelu
import pickle
import csv
from os import listdir

# lista_mfcc_nazwa = odczytywanie_kiszonki('lista_mfcc_nazwa.pkl')
#
# slownik = stworz_slownik(lista_plikow, lista_mfcc_nazwa)
# #acc = accuracy_percent(slownik=slownik, n_splits=5, n_comp=10, n_iter=100, n_init=5, random_state=2)
#
#
# lista_mfcc_nazwa_eval = odczytywanie_kiszonki('lista_mfcc_nazwa_eval.pkl')
#
# y_pred, p_score = co_to_jest_za_cyfra_wg_naszego_modelu(lista_mfcc_nazwa_eval, lista_mfcc_nazwa)
#
# with open('results.csv', newline='', mode='w') as result:
#     result_writer = csv.writer(result, delimiter=',')
#     for file in range(0, len(lista_mfcc_nazwa_eval)):
#         result_writer.writerow([lista_mfcc_nazwa_eval[file][1], y_pred[file],p_score[file]])
#     result.close()


from eval import evaluate

# res_dic=load_results("Wyniczki")
evaluate(results_fname="results.csv")
a=1

