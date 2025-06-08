# boost_AI_L3
🧠 Détection du diabète avec un réseau de neurones (TensorFlow)
Ce projet implémente un modèle d'apprentissage automatique en Python utilisant TensorFlow et Keras pour prédire si une personne est atteinte de diabète, en se basant sur des données cliniques.

📁 Dataset utilisé
Le jeu de données provient de la base Pima Indians Diabetes (UCI Machine Learning Repository), contenant 768 exemples et 8 caractéristiques médicales.

Colonnes :

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

Outcome (0 = non diabétique, 1 = diabétique)

Le dataset est automatiquement chargé depuis :
https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

⚙️ Fonctionnalités
Téléchargement automatique des données

Normalisation avec StandardScaler

Séparation des données (80% entraînement / 20% test)

Réseau de neurones avec 3 couches :

1 couche cachée (10 neurones, ReLU)

1 couche cachée (6 neurones, ReLU)

1 couche de sortie (1 neurone, Sigmoïde)

Entraînement du modèle

Évaluation sur jeu de test

Prédiction personnalisée

🖥️ Installation
Assurez-vous d’avoir Python 3.8+ installé.

Installez les dépendances nécessaires :

bash:
pip install tensorflow pandas scikit-learn numpy
🚀 Exécution
Lancez simplement le script Python :

bash:
python diabetes_detection.py
📊 Exemple de sortie
python-repl:

Epoch 1/100
...
Précision sur le test : 0.77
Probabilité d'être diabétique : 0.81
Classe prédite : 1
🔮 Exemple d'entrée pour prédiction
python:

example = np.array([[6,148,72,35,0,33.6,0.627,50]])
