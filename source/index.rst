.. Owner documentation master file, created by
   sphinx-quickstart on Thu Dec  5 23:15:50 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to darija_data_preprocessing's documentation !
======================================================

Darija Data Preprocessing is a part of the Darija Speech Recognition project, designed to collect, clean, and transform   
raw speech data for model training. This repository handles the collection and preprocessing of data from YouTube,   
improving alignment and preparing it for training the speech recognition model.  

Features
========
1. Data Collecting And Transforming :  
collect.py: Downloads audio files from YouTube, converts their sampling rate to 16 kHz,  
and splits them into chunks based on the transcription timestamps provided by the video.   
Each chunk is associated with its corresponding transcription, and the downloaded chunks   
are stored in a dataset folder within the same project repository.  

2. Data Cleaning :  
delete_extremities.py: Removes the extremities of each full audio to reduce misalignments.  
delete_long_audios.py: Deletes audio chunks longer than 6 seconds to facilitate model training.  
one_word_audios.py: Removes audio chunks containing only one word, as they are prone to misalignment.  
remove_0_sec_audios.py: Deletes audio chunks with 0 seconds duration, which are typically disaligned.  
remove_music_audios.py: Filters out audio chunks with background music, as it is considered noise for the model during training.  

Installation
============
Clone the repository:   
git clone https://github.com/AnassBe34/darija_data_preprocessing.git  

Authors  
=======
This project is part of the Darija Speech Recognition initiative and is 
maintained by Anass Benamara and Hossam Tabsissi.

Contact
=======
anassbenamara8@gmail.com  
hossam.tab84@gmail.com


.. toctree::
   :maxdepth: 2
   :caption: Contents:

Documentation/Scripts/1_Project.rst
Documentation/Scripts/2_Equipe.rst
Documentation/Scripts/3_OCR.rst
Documentation/Scripts/4_Prétraitement.rst
Documentation/Scripts/5_NER_Models.rst
Documentation/Scripts/6_Labellisation.rst
Documentation/Scripts/7_Training.rst
Documentation/Scripts/8_Inference.rst
Documentation/Scripts/9_LLM_Models.rst
Documentation/Scripts/10_perspectives.rst 