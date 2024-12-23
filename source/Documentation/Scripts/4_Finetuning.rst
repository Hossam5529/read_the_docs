IV-Fine-tuning
=============

4.1 Chargement et Préparation des Données :
---------------------------------------------
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
   
Detail :
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


- 4.2 Exemples de reconnaissance d'entités nommées
------------------------------------------------

.. figure:: /Documentation/Images/NER.png
   :width: 80%
   :align: center
   :alt: Alternative text for the image
   :name: NER MODEL

Certains des exemples courants d'un catégorisation d'entité sont: 

 - Apple     : est étiqueté ORG (Organisation) et surligné en rouge.
 - today     : est étiqueté DATE et surligné en rose.
 - Second    : est étiqueté QUANTITÉ et surligné en vert.
 - iPhone SE : est étiqueté COMM (Produit commercial) et surligné en bleu.
 - 4.7-inch  : est étiqueté QUANTITÉ et surligné en vert.

Ambiguïté dans la reconnaissance d'entité nommée 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

La catégorie à laquelle appartient un terme est intuitivement assez claire pour les êtres humains. Cependant, ce n'est pas le cas des ordinateurs,ils rencontrent des problèmes de classification. Par example:
Manchester City (Organisation) a remporté le trophée de la Premier League alors que dans la phrase suivante, l'organisation est utilisée différemment. Manchester City (Localisation) était une centrale électrique textile et industrielle.
Votre modèle NER a besoin données d'entraînement mener avec précision extraction d'entité et classement. Si vous entraînez votre modèle sur l'anglais shakespearien, il va sans dire qu'il ne pourra pas déchiffrer Instagram.

3.Différentes approches NER
----------------------------

L'objectif premier d'un Modèle NER consiste à étiqueter des entités dans des documents texte et à les catégoriser. Les trois approches suivantes sont généralement utilisées à cette fin. Cependant, vous pouvez également choisir de combiner une ou plusieurs méthodes.
Les différentes approches pour créer des systèmes NER sont :

Systèmes basés sur un dictionnaire 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le système basé sur un dictionnaire est peut-être l'approche NER la plus simple et la plus fondamentale. Il utilisera un dictionnaire avec de nombreux mots, des synonymes et une collection de vocabulaire. Le système vérifiera si une entité particulière présente dans le texte est également disponible dans le vocabulaire. En utilisant un algorithme de mise en correspondance de chaînes, une vérification croisée des entités est effectuée.
Un inconvénient de l'utilisation de cette approche est qu'il est nécessaire de mettre à jour constamment l'ensemble de données de vocabulaire pour le fonctionnement efficace du modèle NER.

Systèmes basés sur des règles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans cette approche, les informations sont extraites sur la base d'un ensemble de règles prédéfinies. Il existe deux principaux ensembles de règles utilisées,

- Règles basées sur des modèles : Comme son nom l'indique, une règle basée sur un modèle suit un modèle morphologique ou une chaîne de mots utilisée dans le document.

- Règles basées sur le contexte : Les règles contextuelles dépendent de la signification ou du contexte du mot dans le document.

Systèmes basés sur l'apprentissage automatique
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans les systèmes basés sur l'apprentissage automatique, la modélisation statistique est utilisée pour détecter les entités. Une représentation basée sur les caractéristiques du document texte est utilisée dans cette approche. Vous pouvez surmonter plusieurs inconvénients des deux premières approches puisque le modèle peut reconnaître types d'entités malgré de légères variations dans leur orthographe.

L'apprentissage en profondeur
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Les méthodes d'apprentissage en profondeur pour NER exploitent la puissance des réseaux de neurones tels que les RNN et les transformateurs pour comprendre les dépendances de texte à long terme. Le principal avantage de l’utilisation de ces méthodes est qu’elles sont bien adaptées aux tâches NER à grande échelle avec des données d’entraînement abondantes.
De plus, ils peuvent apprendre des modèles et des fonctionnalités complexes à partir des données elles-mêmes, éliminant ainsi le besoin de formation manuelle. Mais il y a un piège. Ces méthodes nécessitent une grande puissance de calcul pour la formation et le déploiement.

Méthodes hybrides
~~~~~~~~~~~~~~~~~~

Ces méthodes combinent des approches telles que l'apprentissage basé sur des règles, statistique et automatique pour extraire des entités nommées. L’objectif est de combiner les atouts de chaque méthode tout en minimisant leurs faiblesses. L’avantage de l’utilisation de méthodes hybrides est la flexibilité que vous obtenez en fusionnant plusieurs techniques grâce auxquelles vous pouvez extraire des entités de diverses sources de données.
Cependant, il est possible que ces méthodes finissent par devenir beaucoup plus complexes que les méthodes à approche unique, car lorsque vous fusionnez plusieurs approches, le flux de travail peut devenir confus.

NER Models Benchmarking
=========================
Nous avons fait une comparaison entre différents grands modèles de langage, nous avons cité différents modèles en utilisant Hugging Face et LM Studio. 

.. note:: 
   - il faut préparer les données pour chaque modèle pour le Finetuning, ça prend beaucoup de temps et chaque modèle se caractérise par un type des données d'entrée.
   C'est pour cela nous avons utiliser la partie Spaces sur Hugging face.

1.Magorshunov/layoutlm-invoices 
--------------------------------
.. figure:: /Documentation/Images/magorshunov-layoutlm-invoice.png
   :width: 80%
   :align: center
   :alt: Alternative text for the image
   :name: LLM MODEL 

.. note:: 
   - Vous pouvez essayer ce modèle en cliquant `ici <https://huggingface.co/spaces/shalinig/magorshunov-layoutlm-invoices>`_.
2.Faisalraza/layoutlm-invoices 
--------------------------------
.. figure:: /Documentation/Images/faisalraza-layoutlm-invoices.png
   :width: 80%
   :align: center
   :alt: Alternative text for the image
   :name: LLM MODEL 

.. note:: 
   - Vous pouvez essayer ce modèle en cliquant `ici <https://huggingface.co/spaces/Anushk24/faisalraza-layoutlm-invoices>`_.

3.Impira/layoutlm-invoices 
---------------------------
.. figure:: /Documentation/Images/impira-layoutlm-invoices.png
   :width: 80%
   :align: center
   :alt: Alternative text for the image
   :name: LLM MODEL 

.. note:: 
   - Vous pouvez essayer ce modèle en cliquant `ici <https://huggingface.co/spaces/udayzee05/impira-layoutlm-invoices>`_.

4.Invoice header extraction with Donut 
---------------------------------------
.. figure:: /Documentation/Images/donut.png
   :width: 80%
   :align: center
   :alt: Alternative text for the image
   :name: LLM MODEL 

.. note:: 
   - Vous pouvez essayer ce modèle en cliquant `ici <https://huggingface.co/spaces/to-be/invoice_document_headers_extraction_with_donut>`_.

5.Gemini application  
---------------------------------------
.. figure:: /Documentation/Images/gemini.png
   :width: 80%
   :align: center
   :alt: Alternative text for the image
   :name: LLM MODEL 

.. note:: 
   - Vous pouvez essayer ce modèle en cliquant `ici <https://huggingface.co/spaces/pc-17/invoice_extraction>`_.

6.Generative AI / invoice reader
--------------------------------------
.. figure:: /Documentation/Images/generativeAI.png
   :width: 80%
   :align: center
   :alt: Alternative text for the image
   :name: LLM MODEL 

.. note:: 
   - Vous pouvez essayer ce modèle en cliquant `ici <https://huggingface.co/spaces/niladridutta/genai_based_invoice_reader>`_.

7.Invoice Information Extraction using LayoutLMv3 model
----------------------------------------------------------
.. figure:: /Documentation/Images/layoutlmv3.png
   :width: 80%
   :align: center
   :alt: Alternative text for the image
   :name: LLM MODEL 

.. note:: 
   - Vous pouvez essayer ce modèle en cliquant `ici <https://huggingface.co/spaces/Theivaprakasham/layoutlmv3_invoice>`_.


Nous avons réalisé une analyse comparative approfondie de plusieurs modèles de langage de grande envergure (LLM) pour l'extraction de texte à partir de documents. Notre évaluation s'est principalement concentrée sur deux critères essentiels : le temps d'inférence requis par chaque modèle et le poids, ou la taille, de ces modèles. En examinant attentivement ces aspects, nous avons pu classer ces modèles en fonction de leur performance et de leur efficacité dans le contexte de l'extraction de texte. Cette classification nous a fourni des insights précieux sur les forces et les faiblesses de chaque modèle, nous permettant ainsi de prendre des décisions éclairées quant à leur utilisation dans divers scénarios d'application.

Voici une video qui vous aidera à trouver et essayer les NER modèles

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/M1cMBA6R95Y" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>




