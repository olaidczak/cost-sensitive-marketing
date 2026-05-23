# Cost-Sensitive Predictive Modeling

## Plan projektu (PROSZE PRZECZYTAC!!!!!!!!!!!)
ja uzywalam Python 3.13.7

1. EDA
    - dwie zmienne są zduplikowane: V175 = V32, V416 = V160
    
2. Zmniejszenie predyktorów z 500 do ok. 30? 
    - to ma odrzucić zupełnie niepotrzebne predyktory i wyłonić kandydatów których uzyc w modelu.
    - Do tego celu uzylam połączenia 4 metod: Boruta, Anova f statistic, mutual information i  auc per variable. Są one agregowane i dostajemy [`feature_importance_rank.csv`](/src/feature_selection/feature_importance_rank.csv)
    - wzięłam top 40 predyktorów z `feature_importance_rank.csv`
    - Na podstawie analiz macierzy korelacji i wyników `feature_importance_rank.csv` dodatkowo usunęłam zmienne z wysoką korelacją (usuwalam tą zmienną, która była nizej w rankingu) - to wszytsko w [`eda.ipynb`](/src/feature_selection/eda.ipynb)
    - mamy 27 predyktorów (zapisane w pliku [`selected_features.txt`](src/feature_selection/selected_features.txt))
    - chyba najwazniejsze predyktory to ['V11', 'V176', 'V191', 'V255', 'V309']. Zamiast V11 trzeba tez sprawdzic jak zachowuja sie V313 i V224.

3. Kroswalidacja w celu selekcji modelu
    - jako metryka uzywamy nasz [`custom_score`](/src/model/utils.py)
    - klasyfikujemy jako jeden gdy $p > 1/3$
    - oprócz cv ze wzgledu na rodzaje modeli i hiperaparametry wewnątrz tej samej cv trzeba przeprowadzić dalszą część feature selection (jakis forward selection, albo tez popatrzec jak L1 przy duzej lamda, itp itd)
    - no ale to moze najpierw zrob zwykla cv, a potem ja zmodyfikuje zeby tez byla ta feature selection? nwm taka mialam wizje
    - jakas wstepna wersja tego co mialam na mysli jest w [`demo.ipynb`](/src/model/demo.ipynb), ale to tylko zrobilam zebys wiedzial jaki mialam pomysl. mozesz to wywalic i robic zupelnie po swojemu!!!!!!!!

## Repository structure
```
├── data/
│    ├── x_test.txt  
│    ├── x_train.txt  
│    └── y_train.txt 
└── src/
     ├── feature_selection/
     └── model/
```
