IV-Fine-tuning
=============

4.1 Chargement et Préparation des Données de l'entrainement :
------------------------------------------------------------------
Cette étape prépare les données d'entraînement pour les rendre compatibles avec la bibliothèque Hugging Face Datasets, 
essentielle pour travailler avec des modèles comme Wav2Vec2.

.. code-block:: python
   
   import pandas as pd
   from datasets import Dataset, Audio

   file_path = r"C:\Users\ASUS\Desktop\DARIJA_SPEECH_RECOGNITION\Data Preprocessing\data_organization\train.txt" 

   train_data = pd.read_csv(file_path, sep="|", header=0)

   train_data.rename(columns={'path': 'audio', 'transcript': 'text'}, inplace=True)

   train_data = train_data.iloc[:, :2]

   train_data_hf = Dataset.from_pandas(train_data)

   train_data_hf = train_data_hf.cast_column("audio", Audio())

   print(train_data_hf.column_names)  
   print(train_data_hf.features)     

.. code-block:: python

   ['audio', 'text']
   {'audio': Audio(sampling_rate=None, mono=True, decode=True, id=None), 'text': Value(dtype='string', id=None)}

Détail :
~~~~~~~~~~~

- 1.Chargement des données :

    - Le fichier train.txt contient les chemins des fichiers audio et leurs transcriptions associées.
    - Les données sont chargées dans un DataFrame Pandas en séparant les colonnes avec le délimiteur |.

- 2.Renommage des colonnes :

    - Les colonnes path et transcript sont renommées respectivement en audio et text pour correspondre aux formats attendus.

- 3.Conversion en Dataset Hugging Face :

    - La bibliothèque Hugging Face est utilisée pour convertir les données dans un format compatible avec les modèles.
    - La colonne audio est spécifiée comme contenant des fichiers audio en la "castant" au type Audio().

- 4.Affichage des métadonnées :

    - column_names : Affiche les colonnes (audio, text).
    - features : Montre les types de données (Audio pour les fichiers audio, String pour les transcriptions).


4.2 Chargement et Préparation des Données de Test :
-----------------------------------------------------
Cette étape prépare les données de test de la même manière que les données d'entraînement, 
afin qu'elles soient compatibles avec le modèle et le pipeline d'évaluation.

.. code-block:: python
   import pandas as pd
   from datasets import Dataset, Audio

   # Chemin vers le fichier de données de test
   file_path = r"C:\Users\ASUS\Desktop\DARIJA_SPEECH_RECOGNITION\Data Preprocessing\data_organization\test.txt" 

   # Chargement des données dans un DataFrame Pandas
   test_data = pd.read_csv(file_path, sep="|", header=0)

   # Renommer les colonnes pour correspondre aux noms attendus par Hugging Face
   test_data.rename(columns={'path': 'audio', 'transcript': 'text'}, inplace=True)

   # Conversion du DataFrame Pandas en un dataset Hugging Face
   test_data_hf = Dataset.from_pandas(test_data)

   # Convertir la colonne "audio" au format audio de Hugging Face
   test_data_hf = test_data_hf.cast_column("audio", Audio())

   # Afficher les premières lignes des données
   print(test_data.head())

Détail :
~~~~~~~~~~~

- 1.Chargement des données de test :

   - Le fichier test.txt contient les chemins des fichiers audio de test et leurs transcriptions.
   - Les colonnes sont séparées par le délimiteur |.

- 2.Renommage des colonnes :

   - Les colonnes path (chemin des fichiers audio) et transcript (transcriptions) sont renommées en audio et text.

- 3.Conversion en Dataset Hugging Face :

   - Les données sont converties en un dataset Hugging Face.
   - La colonne audio est castée au type Audio() pour permettre une gestion efficace des fichiers audio.

- 4.Affichage des données :

    - La commande print(test_data.head()) permet de visualiser les premières lignes du DataFrame pour vérifier que les données ont été correctement chargées et formatées.

4.3 Extraction du Vocabulaire Unique :
--------------------------------------

Dans cette étape, vous extrayez les caractères uniques présents dans les transcriptions des données d'entraînement et de test. 
Ce vocabulaire est utilisé pour définir l'ensemble des symboles que le modèle doit apprendre à reconnaître.

.. code-block:: python

   def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

   # Extraire le vocabulaire des données d'entraînement
   vocab_train = train_data_hf.map(
      extract_all_chars, 
      batched=True, 
      batch_size=-1, 
      keep_in_memory=True, 
      remove_columns=train_data_hf.column_names
   )

   # Extraire le vocabulaire des données de test
   vocab_test = test_data_hf.map(
      extract_all_chars, 
      batched=True, 
      batch_size=-1, 
      keep_in_memory=True, 
      remove_columns=test_data_hf.column_names
   )

   vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
   vocab_dict = {v: k for k, v in enumerate(vocab_list)}
   vocab_dict

.. code-block:: python

   {'ث': 0,
   'ء': 1,
   'و': 2,
   'ز': 3,
   '7': 4,
   'ئ': 5,
   'ى': 6,
   'ش': 7,
   'ت': 8,
   '8': 9,
   '2': 10,
   'ب': 11,
   ' ': 12,
   'ط': 13,
   'س': 14,
   'ا': 15,
   'ظ': 16,
   '0': 17,
   'ح': 18,
   'ع': 19,
   '3': 20,
   '9': 21,
   'ذ': 22,
   'د': 23,
   'ج': 24,
   ...
   '5': 37,
   'ف': 38,
   'ل': 39,
   'غ': 40,
   'ك': 41}

Détail :
~~~~~~~~~~
- 1.Fonction extract_all_chars :

    - Prend un batch de données (batch) en entrée.
   - Combine toutes les transcriptions (batch["text"]) en une seule chaîne.
   - Identifie les caractères uniques à l'aide de set() et les convertit en liste.

- 2.Application de la fonction :

    - La fonction est appliquée à toutes les données d'entraînement et de test à l'aide de la méthode map() de Hugging Face Datasets.
    - Paramètres importants :
        - batched=True : La fonction est appliquée à des lots de données, et non à des exemples individuels.
        - batch_size=-1 : Le lot contient toutes les données (calcul global).
        - keep_in_memory=True : Garde les données en mémoire pour un traitement rapide.
        - remove_columns=train_data_hf.column_names : Supprime les colonnes d'origine pour ne garder que les résultats de la fonction.

4.4  Création du Vocabulaire et du Processeur pour Wav2Vec2 :
-------------------------------------------------------------------
Dans cette étape, le vocabulaire extrait est sauvegardé dans un fichier JSON. Ensuite, 
un tokenizer et un processeur sont configurés pour préparer les données d'entrée au modèle Wav2Vec2.

4.4.1 Sauvegarde du vocabulaire en JSON :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import json
   with open('vocab.json', 'w', encoding='utf-8') as vocab_file:
      json.dump(vocab_dict, vocab_file, ensure_ascii=False, indent=4)

- vocab_dict : Dictionnaire contenant le vocabulaire, construit à partir des caractères uniques extraits.
- Paramètres importants :
    - ensure_ascii=False : Permet de sauvegarder correctement les caractères non latins (par exemple, en arabe).
    - indent=4 : Ajoute une indentation pour rendre le fichier JSON lisible.

4.4.2 Création du tokenizer :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from transformers import Wav2Vec2CTCTokenizer
   tokenizer = Wav2Vec2CTCTokenizer(
      "./vocab.json", 
      unk_token="[UNK]", 
      pad_token="[PAD]", 
      word_delimiter_token="|"
   )

- **Wav2Vec2CTCTokenizer :**

     - Convertit les caractères des transcriptions en séquences numériques.
     - unk_token="[UNK]" : Spécifie le jeton utilisé pour les caractères inconnus.
     - pad_token="[PAD]" : Définit le jeton pour le padding.
     - word_delimiter_token="|" : Séparateur pour délimiter les mots.

4.4.3 Création du feature extractor :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from transformers import Wav2Vec2FeatureExtractor
   feature_extractor = Wav2Vec2FeatureExtractor(
      feature_size=1, 
      sampling_rate=16000, 
      padding_value=0.0, 
      do_normalize=True, 
      return_attention_mask=True
   )

- **Wav2Vec2FeatureExtractor :**

      - Prépare les données audio brutes pour l'entrée dans le modèle.
      - Paramètres importants :
         - feature_size=1 : Spécifie la taille des caractéristiques (mono signal).
         - sampling_rate=16000 : Fréquence d'échantillonnage des fichiers audio.
         - do_normalize=True : Normalise les valeurs d'amplitude de l'audio.
         - return_attention_mask=True : Retourne un masque d'attention pour ignorer les portions padées.

4.4.4 . Création du processeur
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   from transformers import Wav2Vec2Processor
   processor = Wav2Vec2Processor(
      feature_extractor=feature_extractor, 
      tokenizer=tokenizer
)

- **Wav2Vec2Processor :**

    - Combine le tokenizer et le feature extractor dans une seule entité.
    - Utilisé pour préparer les données audio et textuelles à l'entrée du modèle.

4.5  Préparation des Données pour l'Entrée du Modèle :
--------------------------------------------------------

Dans cette étape, vous utilisez le processeur défini précédemment pour transformer les données brutes (audio et texte) en un 
format directement utilisable par le modèle Wav2Vec2. Cela inclut la conversion des signaux audio en représentations
numériques et des transcriptions en séquences d'indices.

.. code-block:: python

   def prepare_dataset(batch, processor=processor):
      # Extraire l'audio et le texte
      audio = batch["audio"]
      text = batch["text"]

      # Transformer les données audio en valeurs d'entrée pour le modèle
      batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]

      # Transformer le texte en étiquettes pour l'entraînement
      with processor.as_target_processor():
         batch["labels"] = processor(text).input_ids

      return batch

- Entrées :

    - batch : Une ligne du dataset contenant l'audio (audio) et la transcription (text).
    - processor : Le processeur défini précédemment.

- Traitement audio :

    - Le processeur extrait des valeurs d'entrée (input_values) à partir des signaux audio en utilisant une fréquence d'échantillonnage de 16 kHz.

- Traitement texte :

    - Les transcriptions sont converties en indices numériques (input_ids) à l'aide du tokenizer du processeur.

4.6  Création d'un Data Collator pour le Fine-Tuning :
------------------------------------------------------

Le data collator est une classe qui gère le processus de mise en lot (batching) des données tout en appliquant un 
padding dynamique aux entrées et aux étiquettes. Cette étape est essentielle pour garantir que les données sont correctement
 alignées lorsqu'elles sont passées au modèle.

.. code-block:: python

   import torch
   from dataclasses import dataclass, field
   from typing import Any, Dict, List, Optional, Union

   @dataclass
   class DataCollatorCTCWithPadding:
    
      processor: Wav2Vec2Processor
      padding: Union[bool, str] = True
      max_length: Optional[int] = None
      max_length_labels: Optional[int] = None
      pad_to_multiple_of: Optional[int] = None
      pad_to_multiple_of_labels: Optional[int] = None

      def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
         # Séparer les entrées et les labels car leurs longueurs diffèrent
         input_features = [{"input_values": feature["input_values"]} for feature in features]
         label_features = [{"input_ids": feature["labels"]} for feature in features]

         # Appliquer le padding aux entrées
         batch = self.processor.pad(
               input_features,
               padding=self.padding,
               max_length=self.max_length,
               pad_to_multiple_of=self.pad_to_multiple_of,
               return_tensors="pt",  # Retourner des tenseurs PyTorch
         )

         # Appliquer le padding aux étiquettes
         with self.processor.as_target_processor():
               labels_batch = self.processor.pad(
                  label_features,
                  padding=self.padding,
                  max_length=self.max_length_labels,
                  pad_to_multiple_of=self.pad_to_multiple_of_labels,
                  return_tensors="pt",
               )

         # Remplacer le padding par -100 pour ignorer ces positions dans le calcul de la perte
         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
         batch["labels"] = labels

         return batch

- **Détails de la Classe DataCollatorCTCWithPadding**

    - **Attributs :**
        - processor : Le processeur défini précédemment, qui gère le padding pour les entrées et les étiquettes.
        - padding : Stratégie de padding (par exemple, "longest" ou "max_length").
        - max_length : Longueur maximale des entrées après padding.
        - max_length_labels : Longueur maximale des étiquettes après padding.
        - pad_to_multiple_of : Alignement des entrées sur une longueur multiple (facilite l'utilisation des Tensor Cores).
        - pad_to_multiple_of_labels : Idem pour les étiquettes.

    - **Méthode __call__ :**
        - Séparation des données :
            - Les entrées (input_values) et les étiquettes (labels) sont séparées car leurs longueurs peuvent différer.
        - Padding des entrées :
            - La méthode processor.pad() est utilisée pour ajuster les longueurs des séquences d'entrée.
        - Padding des étiquettes :
            - Les étiquettes sont également padées avec la méthode du processeur.
        - Remplacement du padding :
            - Les valeurs de padding sont remplacées par -100, une valeur spéciale utilisée pour ignorer ces positions dans le calcul de la perte (CTC Loss).

    - **Retour :**
        - Un dictionnaire contenant les données d'entrée (input_values) et les étiquettes (labels) après padding.

4.7 Calcul des Métriques de Performance :
-------------------------------------------
Cette étape définit une fonction pour évaluer les performances du modèle de transcription. La métrique principale utilisée ici est le Word Error Rate (WER), 
une mesure standard pour évaluer les systèmes de reconnaissance vocale.

.. code-block:: python

   def compute_metrics(pred):
      # Obtenir les logits des prédictions
      pred_logits = pred.predictions

      # Convertir les logits en indices de prédiction
      pred_ids = np.argmax(pred_logits, axis=-1)

      # Remplacer les labels -100 par l'ID du token [PAD] (padding)
      pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

      # Décoder les prédictions en texte
      pred_str = processor.batch_decode(pred_ids)

      # Décoder les labels (sans regrouper les tokens)
      label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

      # Calculer le Word Error Rate
      wer = wer_metric.compute(predictions=pred_str, references=label_str)

      return {"wer": wer}

- Détails de la Fonction compute_metrics

    - Logits des prédictions :
        - Les sorties du modèle (pred.predictions) sont des logits, qui doivent être convertis en indices.

    - Conversion des logits en indices :
        - La fonction np.argmax est utilisée pour sélectionner l'indice avec la probabilité la plus élevée pour chaque position dans la séquence.

    - Gestion des labels -100 :
        - Les valeurs -100 (utilisées pour ignorer les positions padées) dans pred.label_ids sont remplacées par l'identifiant du token de padding (pad_token_id).

    - Décodage des prédictions et des labels :
        - La méthode batch_decode du tokenizer reconvertit les indices en texte lisible.
        - Les prédictions (pred_str) et les références (label_str) sont obtenues.
    - Calcul du Word Error Rate (WER) :
        - La fonction wer_metric.compute compare les transcriptions prédictes avec les transcriptions de référence pour calculer le taux d'erreurs :
        
      .. figure:: /Documentation/Images/wer.PNG
         :width: 70%
         :align: center
         :alt: Alternative text for the image
         :name: image


    - Retour des résultats :
        - La fonction retourne un dictionnaire contenant la métrique WER : {"wer": wer}.

4.8 Chargement et Configuration du Modèle Wav2Vec2 :
-------------------------------------------------------

Dans cette étape, le modèle Wav2Vec2ForCTC est initialisé à partir d'un modèle pré-entraîné de Hugging Face et personnalisé 
pour l'adaptation aux données spécifiques.

.. code-block:: python

   from transformers import Wav2Vec2ForCTC

   model = Wav2Vec2ForCTC.from_pretrained(
      "facebook/wav2vec2-large-xlsr-53",  # Modèle pré-entraîné Wav2Vec2
      attention_dropout=0.1,             # Dropout dans les mécanismes d'attention
      hidden_dropout=0.1,                # Dropout dans les couches cachées
      feat_proj_dropout=0.0,             # Pas de dropout sur la projection des caractéristiques
      mask_time_prob=0.05,               # Probabilité de masquage temporel pour l'entraînement
      layerdrop=0.1,                     # Dropout entre les couches
      ctc_loss_reduction="mean",         # Moyenne pour le calcul de la perte CTC
      pad_token_id=processor.tokenizer.pad_token_id,  # ID du token de padding
      vocab_size=len(processor.tokenizer)            # Taille du vocabulaire
  )

4.9 Configuration des Arguments d'Entraînement :
--------------------------------------------------
Dans cette étape, vous configurez les paramètres d'entraînement pour le modèle Wav2Vec2ForCTC en utilisant la classe TrainingArguments de Hugging Face. Ces arguments déterminent les hyperparamètres et les réglages 
nécessaires pour l'entraînement et l'évaluation.

.. code-block:: python

   from transformers import TrainingArguments
   training_args = TrainingArguments(
      output_dir=r"C:\Users\ASUS\Desktop\finetuning",  # Dossier pour sauvegarder les checkpoints et logs
      group_by_length=True,                           # Grouper les séquences de longueurs similaires dans un lot
      per_device_train_batch_size=32,                 # Taille du batch pour chaque appareil (GPU/CPU)
      gradient_accumulation_steps=4,                  # Accumuler les gradients sur 4 batches avant une mise à jour
      evaluation_strategy="steps",                    # Évaluer le modèle après un certain nombre de pas
      num_train_epochs=30,                            # Nombre total d'époques d'entraînement
      fp16=True,                                      # Utiliser la précision mixte (FP16) pour accélérer l'entraînement
      save_steps=100,                                 # Sauvegarder les checkpoints tous les 100 pas
      eval_steps=100,                                 # Évaluer le modèle tous les 100 pas
      logging_steps=10,                               # Journaliser les métriques toutes les 10 étapes
      learning_rate=3e-4,                             # Taux d'apprentissage
      warmup_steps=500,                               # Étapes de réchauffement pour le scheduler de LR
      save_total_limit=2,                             # Limiter le nombre de checkpoints sauvegardés à 2
   )

4.9.1 Création du Trainer pour l'Entraînement
----------------------------------------------
La classe Trainer de Hugging Face est utilisée pour orchestrer l'ensemble du processus d'entraînement, d'évaluation et de gestion des données. Elle combine le modèle, les données, les métriques, et les arguments d'entraînement pour offrir 
un pipeline d'entraînement clé en main.

.. code-block:: python 

   from transformers import Trainer
   trainer = Trainer(
      model=model,                          # Modèle Wav2Vec2 pour la tâche CTC
      data_collator=data_collator,          # Collateur de données pour gérer le padding et les lots
      args=training_args,                   # Arguments d'entraînement configurés précédemment
      compute_metrics=compute_metrics,      # Fonction pour calculer les métriques (ex. : WER)
      train_dataset=train_data_hf,          # Dataset d'entraînement
      eval_dataset=test_data_hf,            # Dataset de test/évaluation
      tokenizer=processor.feature_extractor # Tokenizer pour traiter les données audio
   )






