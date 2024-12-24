.. Owner documentation master file, created by
   sphinx-quickstart on Thu Dec  5 23:15:50 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to darija voice transcription's and translation's documentation !
===========================================================================

This project transcribes and translates Darija (Moroccan Arabic) audio into text with two main components:

    - **Audio Transcription Model** : Utilizes the Wav2Vec2-large-XLSR-53 model, a state-of-the-art model for speech recognition, fine-tuned on a Darija Dataset, to transcribe audio into accurate text.
    - **Translation Model** : Leverages a fine-tuned version of Helsinki-NLP/opus-mt-ar-en, trained on the None dataset, to translate the transcriptions from Darija into English.

The repository also includes essential tools for collecting data from YouTube videos, including audio and their corresponding transcriptions based on video timestamps. It offers scripts for cleaning, transforming, and organizing the data to make it suitable for training and fine-tuning the the Wav2Vec2-large-XLSR-53 model.

A simple app is also provided that enables users to upload audio files and receive both transcriptions and translations in a straightforward interface.

 ..

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
   