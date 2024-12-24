.. Owner documentation master file, created by
   sphinx-quickstart on Thu Dec  5 23:15:50 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bienvenue dans la documentation sur la transcription et la traduction des voix en darija !
===============================================================================================

Ce projet transcrit et traduit l'audio en darija (arabe marocain) en texte à l'aide de deux composants principaux :

    - **Modèle de transcription audio :** Utilise le modèle Wav2Vec2-large-XLSR-53, un modèle de pointe pour la reconnaissance vocale, affiné sur un jeu de données en darija, pour transcrire l'audio en texte précis.
    - **Modèle de traduction :** Exploite une version fine-tunée de Helsinki-NLP/opus-mt-ar-en, entraînée sur le jeu de données None, pour traduire les transcriptions du darija vers l'anglais.

Le dépôt inclut également des outils essentiels pour collecter des données à partir de vidéos YouTube, notamment les pistes audio et leurs transcriptions correspondantes basées sur les horodatages des vidéos. Il propose des scripts pour nettoyer, transformer et organiser les données afin de les rendre adaptées à l'entraînement et au fine-tuning du modèle Wav2Vec2-large-XLSR-53.

Une application simple est également fournie, permettant aux utilisateurs de télécharger des fichiers audio et de recevoir à la fois les transcriptions et les traductions via une interface intuitive.

.. toctree::
   :maxdepth: 2
   :caption: Table des matières:

   Documentation/Scripts/1_Introduction.rst
   Documentation/Scripts/2_Equipe.rst
   Documentation/Scripts/3_Data_preprocessing.rst
   Documentation/Scripts/4_Finetuning.rst
   Documentation/Scripts/5_Translation.rst
   Documentation/Scripts/6_Application_projet.rst
   Documentation/Scripts/7_perspectives.rst
   