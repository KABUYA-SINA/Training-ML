import os
import warnings

# Désactiver les opérations personnalisées oneDNN et réduire les messages de débogage
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Supprimer les avertissements
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow_model_remediation import min_diff


#logits et labels
logits = tf.random.normal([5, 10])
labels = tf.constant([3, 1, 4, 0, 2], dtype=tf.int64)

# Utilisation de tf.keras.losses pour éviter la dépréciation
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss = loss_fn(labels, logits)

# Importation des données
acs_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/acsincome_raw_2018.csv")

# Le PINCP (revenu annuel total de la personne). L'étiquette cible d'un individu sera 1 si le PINCP est > 50000,0, sinon 0.
LABEL_KEY = 'PINCP'
LABEL_THRESHOLD = 50000.0

acs_df[LABEL_KEY] = acs_df[LABEL_KEY].apply(
    lambda income: 1 if income > LABEL_THRESHOLD else 0)


# Préparation des fonctionnalités
features = acs_df.copy()
features.pop(LABEL_KEY)



# Définition du modèle de base
inputs = {}
for name, column in features.items():
    if name != LABEL_KEY:
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=tf.float32)

# Empilez les entrées
def stack_dict(inputs):
    return tf.keras.layers.Concatenate()(list(inputs.values()))

# Concaténer les entrées
x = stack_dict(inputs)


# Normalisation des données
normalizer = tf.keras.layers.Normalization(axis=-1)

# Convertir le DataFrame en un tableau numpy
features_array = features.to_numpy()
normalizer.adapt(features_array)

# Construire le modèle
x = normalizer(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Créer le modèle de base
base_model = tf.keras.Model(inputs, outputs)


# Configurer le modèle
METRICS = [
  tf.keras.metrics.BinaryAccuracy(name='accuracy'),
  tf.keras.metrics.AUC(name='auc'),
]

base_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=METRICS
)


# Convertir en tf.data.Dataset
def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop(LABEL_KEY)
    dataset = tf.data.Dataset.from_tensor_slices((
        {name: tf.cast(values.values, tf.float32) for name, values in dataframe.items()},
        tf.cast(labels.values, tf.float32)
    ))
    return dataset


# Séparer l'ensemble de données en train/test
RANDOM_STATE = 200
BATCH_SIZE = 100
EPOCHS = 10

acs_train_df, acs_test_df = train_test_split(acs_df, test_size=0.2, random_state=RANDOM_STATE)

acs_train_ds = dataframe_to_dataset(acs_train_df).batch(BATCH_SIZE)
acs_test_ds = dataframe_to_dataset(acs_test_df).batch(BATCH_SIZE)

# Entraîner le modèle
base_model.fit(acs_train_ds, epochs=EPOCHS)


# Évaluer le modèle
base_model.evaluate(acs_test_ds, batch_size=BATCH_SIZE)

# Prédictions pour l'équité
base_model_predictions = base_model.predict(acs_test_ds, batch_size=BATCH_SIZE)
base_model_predictions = (base_model_predictions > 0.5).astype(int)  # Convertir en 0 ou 1

# Ajout des prédictions dans le DataFrame de test
acs_test_df['PRED'] = base_model_predictions


# Calcul des indicateurs d'équité.
SENSITIVE_ATTRIBUTE_KEY = 'SEX'
SENSITIVE_ATTRIBUTE_VALUES = {1.0: "Male", 2.0: "Female"}
# PREDICTION_KEY = 'PRED'

# Remplacer les valeurs d'attributs sensibles
acs_test_df[SENSITIVE_ATTRIBUTE_KEY].replace(SENSITIVE_ATTRIBUTE_VALUES, inplace=True)

# Calculer les métriques de performance (accuracy, AUC, etc.)
accuracy = accuracy_score(acs_test_df[LABEL_KEY], acs_test_df['PRED'])
roc_auc = roc_auc_score(acs_test_df[LABEL_KEY], base_model_predictions)

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(acs_test_df[LABEL_KEY], acs_test_df['PRED'])

# Afficher les résultats
print(f"Accuracy: {accuracy}")
print(f"AUC: {roc_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")
print("\nClassification Report:")
print(classification_report(acs_test_df[LABEL_KEY], acs_test_df['PRED']))



# Tâche 1: Cerner les préoccupations en matière d'équité

# Calculer l'équité (par exemple, les scores pour les groupes sensibles)
sensitive_groups = acs_test_df.groupby(SENSITIVE_ATTRIBUTE_KEY)

print("\nFairness Metrics by Group:")

for group, group_data in sensitive_groups:
    group_accuracy = accuracy_score(group_data[LABEL_KEY], group_data['PRED'])
    group_auc = roc_auc_score(group_data[LABEL_KEY], group_data['PRED'])
    group_conf_matrix = confusion_matrix(group_data[LABEL_KEY], group_data['PRED'])
    
    print(f"\nGroup: {group}")
    print(f"Accuracy: {group_accuracy:.4f}")
    print(f"AUC: {group_auc:.4f}")
    print(f"Confusion Matrix:\n{group_conf_matrix}")
    print("Classification Report:")
    print(classification_report(group_data[LABEL_KEY], group_data['PRED']))

# Visualisation de la distribution des prédictions par groupe
plt.figure(figsize=(10, 6))
sns.boxplot(x=SENSITIVE_ATTRIBUTE_KEY, y='PRED', data=acs_test_df)
plt.title('Distribution of Predictions by Group')
plt.show()

# Visualisation des taux de vrais positifs et faux positifs par groupe
def plot_roc_by_group(y_true, y_pred, groups):
    plt.figure(figsize=(10, 8))
    for group in groups.unique():
        mask = groups == group
        fpr, tpr, _ = roc_curve(y_true[mask], y_pred[mask])
        auc = roc_auc_score(y_true[mask], y_pred[mask])
        plt.plot(fpr, tpr, label=f'{group} (AUC = {auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve by Group')
    plt.legend(loc="lower right")
    plt.show()

plot_roc_by_group(acs_test_df[LABEL_KEY], base_model_predictions, acs_test_df[SENSITIVE_ATTRIBUTE_KEY])




# la question de l'équité
# Tâche 2: Créer des sous-ensembles marqués positivement

sensitive_group_pos = acs_train_df[
    (acs_train_df[SENSITIVE_ATTRIBUTE_KEY] == 2.0) & (acs_train_df[LABEL_KEY] == 1)]
non_sensitive_group_pos = acs_train_df[
    (acs_train_df[SENSITIVE_ATTRIBUTE_KEY] == 1.0) & (acs_train_df[LABEL_KEY] == 1)]

print(len(sensitive_group_pos),
      'positively labeled sensitive group examples')
print(len(non_sensitive_group_pos),
      'positively labeled non-sensitive group examples')