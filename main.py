import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification
from transformers import TrainingArguments, Trainer
# from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Charger les données
df = pd.read_csv('dataset/data.csv')

# Sélectionner les colonnes pertinentes
df = df[['test_case', 'label_gold', 'target_ident']]

# Convertir le label en numérique
df['label_gold'] = df['label_gold'].map({'hateful': 1, 'non-hateful': 0})

# Diviser le jeu de données en entraînement et test
train_texts, test_texts, train_labels, test_labels = train_test_split(df['test_case'], df['label_gold'], test_size=0.2)


tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)

class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = HateSpeechDataset(train_encodings, train_labels)
test_dataset = HateSpeechDataset(test_encodings, test_labels)

# Charger le modèle pré-entrainé
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)

# Définir les paramètres d'entraînement
training_args = TrainingArguments(
    output_dir='./results',          # sorties du modèle
    num_train_epochs=3,              # nombre total d'époques
    per_device_train_batch_size=16,  # taille du batch par GPU/CPU
    per_device_eval_batch_size=64,   # taille du batch pour l'évaluation
    warmup_steps=500,                # nombre de pas d'échauffement 
    weight_decay=0.01,               # taux de décroissance du poids
    logging_dir='./logs',            # répertoire pour les logs
)

# Créer le Trainer et commencer l'entraînement
trainer = Trainer(
    model=model,                         # le modèle à entraîner
    args=training_args,                  # les paramètres d'entraînement
    train_dataset=train_dataset,         # les données d'entraînement
    eval_dataset=test_dataset            # les données de validation
)

trainer.train()


# Faire des prédictions
predictions = trainer.predict(test_dataset)

# Convertir les prédictions en labels
pred_labels = predictions[0].argmax(axis=-1)

# Générer la matrice de confusion
cm = confusion_matrix(test_labels, pred_labels)
print(cm)
