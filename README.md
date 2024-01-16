# Projet personnel - E2

## 1. Présentation du projet

Les maladies cardiaques sont l'une des principales causes de décès dans le monde, et leur prévention précoce est d'une importance capitale pour la santé publique. Dans le cadre de ce projet, j'ai utilisé un notebook Jupyter disponible sur Kaggle comme point de départ pour développer un modèle de prédiction des maladies cardiaques. Mon objectif était d'améliorer les performances du modèle existant en utilisant différentes techniques et approches.

[Lien Kaggle](https://www.kaggle.com/code/elouanguyon/pr-diction-maladies-cardiaques)

IA utilisé: La régression logistique (apprentissage supervisé).

----

## 2. Installation des librairies

Pour installer les librairies nécessaires, vous pouvez utiliser le fichier requirements.txt. Pour cela, utilisez la commande suivante dans votre terminal:

```
pip install -r requirements.txt
```

----

## 3. Amélioration du modèle

Précision à améliorer:
- Précision train: 0.8536585365853658 
- Précision test: 0.824390243902439


#### Etape 1:													
Sélection des colonnes à conserver en fonction de leur corrélation.													
Récupération des seuils au delà de 0.1 à 0.4													
													
Résultat
```python
('Précision train: 0.8575609756097561', ' Précision test: 0.8341463414634146', 'Seuil de corrélation : 0.0')													
('Précision train: 0.8585365853658536', ' Précision test: 0.8390243902439024', 'Seuil de corrélation : 0.1')													
('Précision train: 0.8390243902439024', ' Précision test: 0.8341463414634146', 'Seuil de corrélation : 0.2')													
('Précision train: 0.8526829268292683', ' Précision test: 0.8292682926829268', 'Seuil de corrélation : 0.3')													
('Précision train: 0.7902439024390244', ' Précision test: 0.7463414634146341', 'Seuil de corrélation : 0.4')	

0.824390243902439 à 0.8390243902439024
```
													
### Etape 2:													
Modification des hyper-paramètres:													
De base: Log_Reg = LogisticRegression(max_iter=1000)													
Modifié: Log_Reg = LogisticRegression(max_iter=3000, solver="saga", random_state=10, class_weight= "balanced")													
													
Résultat	
```python
('Précision train: 0.8497560975609756', ' Précision test: 0.8341463414634146', 'Seuil de corrélation : 0.0')													
('Précision train: 0.855609756097561', ' Précision test: 0.8341463414634146', 'Seuil de corrélation : 0.1')													
('Précision train: 0.8497560975609756', ' Précision test: 0.8439024390243902', 'Seuil de corrélation : 0.2')													
('Précision train: 0.8497560975609756', ' Précision test: 0.8292682926829268', 'Seuil de corrélation : 0.3')													
('Précision train: 0.7951219512195122', ' Précision test: 0.7463414634146341', 'Seuil de corrélation : 0.4')

0.8390243902439024 à 0.8439024390243902
```												
													
### Etape 3:													
Normaliser les données des colonnes "RestingBP", "Cholesterol" et "MaxHR" (entre 0 et 1).													
													
Dataframe

| Age | Sex | ChestPainType | RestingBP | Cholesterol | FastingBS | RestingECG | MaxHR | ExerciseAngina | Oldpeak | ST_Slope | ca | thal | HeartDisease |
|-----|-----|---------------|-----------|-------------|-----------|------------|-------|----------------|---------|----------|----|------|--------------|
| 53  |  1  |       0       |    140    |     203     |     1     |     0      |  155  |       1        |   3.1   |    0     |  0 |   3  |      0       |
| 54  |  0  |       2       |    135    |     304     |     1     |     1      |  170  |       0        |   0.0   |    2     |  0 |   2  |      1       |
| 57  |  0  |       0       |    128    |     303     |     0     |     0      |  159  |       0        |   0.0   |    2     |  1 |   2  |      1       |
| 53  |  1  |       2       |    130    |     197     |     1     |     0      |  152  |       0        |   1.2   |    0     |  0 |   2  |      1       |
| 59  |   1 |       0       |    138    |     271     |     0     |     0      |  182  |       0        |   0.0   |    2     |  0 |   2  |      1       |

													
Résultat
```python
('Précision train: 0.8663414634146341', ' Précision test: 0.8439024390243902', 'Seuil de corrélation : 0.0')													
('Précision train: 0.8663414634146341', ' Précision test: 0.848780487804878', 'Seuil de corrélation : 0.1')													
('Précision train: 0.8468292682926829', ' Précision test: 0.8439024390243902', 'Seuil de corrélation : 0.2')													
('Précision train: 0.8390243902439024', ' Précision test: 0.8195121951219512', 'Seuil de corrélation : 0.3')													
('Précision train: 0.7863414634146342', ' Précision test: 0.7414634146341463', 'Seuil de corrélation : 0.4')

0.8439024390243902 à 0.848780487804878
```
----												
													
## 4. Amélioration avec RandomForest:													

```python										
randomforest_classifier= RandomForestClassifier(n_estimators=10)													
score=cross_val_score(randomforest_classifier,x_train,y_train,cv=10)													
score.mean()
```												
													
Résultat
```python											
score = 0.9817073170731707													
```

RandomForest bien meilleur, pourquoi?	

1. Capacité à modéliser des relations non linéaires : Les maladies cardiaques peuvent être influencées par des relations complexes et non linéaires entre les caractéristiques (variables d'entrée) et la présence de la maladie (variable cible). La régression logistique, étant un modèle linéaire, peut avoir du mal à capturer ces relations non linéaires, tandis que Random Forest, en tant que modèle ensembliste non paramétrique, est capable de modéliser des relations plus complexes et flexibles.

2. Gestion des interactions entre les caractéristiques : Les caractéristiques dans les données sur les maladies cardiaques peuvent interagir les unes avec les autres pour influencer la présence de la maladie. Random Forest, en construisant plusieurs arbres de décision indépendants, est capable de prendre en compte ces interactions entre les caractéristiques, ce qui peut améliorer la précision de la prédiction.

3. Robustesse aux valeurs aberrantes et aux données bruitées : Les données médicales peuvent contenir des valeurs aberrantes ou des bruits, ce qui peut affecter la performance des modèles. Random Forest est généralement plus robuste que la régression logistique face à ces données aberrantes et bruitées, car il agrège les prédictions de plusieurs arbres, ce qui atténue l'impact de valeurs aberrantes ou de données bruitées sur les résultats finaux.

4. Gestion de la multicollinéarité : Si vos caractéristiques présentent une certaine corrélation entre elles (multicollinéarité), cela peut affecter les performances de la régression logistique, car elle peut avoir du mal à distinguer l'effet spécifique de chaque caractéristique. Random Forest, en construisant des arbres indépendants, est moins sensible à la multicollinéarité et peut mieux gérer ce type de situation.

**Conclusion**

Random Forest a généralement une meilleure capacité à modéliser des relations non linéaires complexes et peut offrir de bonnes performances prédictives.
Cependant, il peut être plus lent à entraîner et à prédire que la régression logistique, en particulier pour de grandes quantités de données.