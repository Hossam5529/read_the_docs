Prétraitement des données : 
=========================

3.1 Collection des données :
-------------------------------------------
Ce script permet de télécharger des vidéos YouTube au format audio (MP3), de les convertir en WAV, 
puis de découper ces fichiers en segments à partir des transcriptions générées pour chaque vidéo.
Concrètement, le code utilise yt-dlp pour extraire le flux audio, pydub pour la conversion et la découpe,
et la bibliothèque youtube_transcript_api pour récupérer les sous-titres (ici en arabe). 
Chaque segment audio est ensuite enregistré avec un fichier texte correspondant, 
ce qui facilite l'analyse et le traitement ultérieur des extraits sonores et de leurs transcriptions.


3.1.1 Installer les dépendances et les bibliothèques : 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   import yt_dlp as youtube_dl  # Now using yt-dlp for better support
   import os
   from pydub import AudioSegment
   from youtube_transcript_api import YouTubeTranscriptApi
 
3.1.2 Définir la liste des URL :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dans le script, renseignez toutes les URL YouTube que vous souhaitez traiter dans la variable url_list.

.. code-block:: python

   url_list = [
   'https://www.youtube.com/watch?v=N8VbED0CPXc',
   'https://www.youtube.com/watch?v=ikUHqsSZROQ',
   'https://www.youtube.com/watch?v=sEehKE0dhps',
   ]

3.1.3 Exécuter le téléchargement et la conversion : 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pour chaque URL de la liste :

    - Téléchargez la piste audio au format MP3,
    - Modifiez la fréquence d'échantillonnage (16 kHz),
    - Puis convertissez le fichier en WAV.

.. code-block:: python

   def download_audio(yt_url, folder_name, audio_name):
      ydl_opts = {
         'format': 'bestaudio/best',
         'postprocessors': [{
               'key': 'FFmpegExtractAudio',
               'preferredcodec': 'mp3',
               'preferredquality': '192',
         }],
         'outtmpl': f'{folder_name}/{audio_name}', 
      }
      try:
         with youtube_dl.YoutubeDL(ydl_opts) as ydl:
               ydl.download([yt_url])
         print(f"Audio downloaded for {yt_url} and saved as {audio_name} in {folder_name}")
      except Exception as e:
         print(f"An error occurred while downloading audio: {e}")
         
.. code-block:: python

   def convert_to_wav(folder_name, audio_name):
      input_file = f"{folder_name}/{audio_name}.mp3"
      output_file = f"{folder_name}/{audio_name}.wav"
      audio = AudioSegment.from_mp3(input_file)
      audio.export(output_file, format="wav")

      if os.path.exists(output_file):
         os.remove(input_file)
         print(f"Conversion complete! '{input_file}' has been replaced by '{output_file}'.")
      else:
         print("Conversion failed; .wav file was not created.")

3.1.4 Récupérer les sous-titres : 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
À l'aide de la fonction get_transcriptions(), récupérez les transcriptions de la vidéo (ici, spécifiées en arabe).

.. code-block:: python

 def get_transcriptions(video_id) :
    transcription = YouTubeTranscriptApi.get_transcript(video_id,  languages=['ar'])
    transcriptions = []
    for element in transcription :
        transcriptions.append(element['text'])
    return transcriptions

3.1.5 Découper l'audio :
~~~~~~~~~~~~~~~~~~~~~~~~~~
- Calculez les intervalles de début et de fin de chaque segment grâce aux fonctions get_starts() et get_ends(),
- Découpez le fichier WAV en plusieurs portions, chacune correspondant à un segment de texte.

.. code-block:: python

 def cut_audio(input_file, output_file, start_time, end_time):
    audio = AudioSegment.from_file(input_file)
    cut_audio = audio[start_time:end_time]
    cut_audio.export(output_file, format = 'wav')

.. code-block:: python

 def get_starts(video_id) :
    transcription = YouTubeTranscriptApi.get_transcript(video_id,  languages=['ar'])
    starts = []
    for element in transcription :
        starts.append(element['start'])
    return starts

.. code-block:: python

   def get_ends(video_id) :
    transcription = YouTubeTranscriptApi.get_transcript(video_id,  languages=['ar'])
    ends = []
    for element in transcription :
        ends.append(element['start'] + element['duration'])
    return ends

3.1.6 Enregistrer les segments et gérer le volume de données :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Sauvegardez chaque portion audio dans un fichier WAV distinct dans un sous-répertoire (par exemple, dataset/dataset_1/audio_chunk_XXXX.wav),
- Créez un fichier texte associé pour y stocker la transcription de ce segment.
- Lorsque le script atteint 20 000 segments, il crée automatiquement un nouveau dossier (p. ex. dataset_2) pour stocker les parties suivantes.
- Enfin, à chaque fin de traitement, supprimez le fichier WAV de l’audio complet, maintenant que vous en avez extrait tous les segments utiles.

.. code-block:: python

   def process_videos(url_list):
    global_chunk_index = 20000
    datasets_index = 1
    
    for i, url in enumerate(url_list, start=0):
        folder_name = f"dataset"
        audio_name = f"audio_{i}"
        audio_file = fr'dataset\audio_{i}.mp3'
        transcription_name = f"transcription_{i}"
      
        download_audio(url, folder_name, audio_name)
        
        audio_mp3 = AudioSegment.from_file(audio_file)
        resampled_audio = audio_mp3.set_frame_rate(16000)
        resampled_audio.export(audio_file, format="mp3")

        convert_to_wav(folder_name, audio_name)
        
        input_file = fr'dataset\audio_{i}.wav'
        video_id = get_video_id(url)
        transcriptions = get_transcriptions(video_id)
        starts = get_starts(video_id)
        ends = get_ends(video_id)
        for j in range(len(starts)) :
            output_file = fr'dataset\dataset_{datasets_index}\audio_chunk_{global_chunk_index}.wav'
            start_time = starts[j] * 1000 - 150
            if start_time < 0 :
                start_time += 150
            if j + 1 < len(starts) :
                end_time = starts[j + 1] * 1000 + 150
                output_file = fr'dataset\dataset_{datasets_index}\audio_chunk_{global_chunk_index}.wav'
                text_file = fr'dataset\dataset_{datasets_index}\audio_chunk_{global_chunk_index}.txt'
            else : 
                end_time = ends[j] * 1000
                output_file = fr'dataset\dataset_{datasets_index}\audio_chunk_{global_chunk_index}_video_end.wav'
                text_file = fr'dataset\dataset_{datasets_index}\audio_chunk_{global_chunk_index}_video_end.txt'
            
            cut_audio(input_file, output_file, start_time, end_time)
            f = open(text_file, "x", encoding="utf-8") 
            f.write(transcriptions[j])
            f.close()
            
            global_chunk_index+=1
            
            if global_chunk_index % 20000 == 0 :
                datasets_index+=1
                os.mkdir(fr'dataset\dataset_{datasets_index}')
            
        os.remove(fr'dataset\audio_{i}.wav')


3.2 Nettoyage des données :
------------------------------
3.2.1 Suppression des extrémités:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ce script est conçu pour supprimer les fichiers audio et texte situés aux extrémités d'un ensemble de segments, en suivant une logique spécifique. L'objectif est de nettoyer le dataset en supprimant les 50 fichiers précédents à un fichier audio particulier détecté, tout en s'assurant que seuls les fichiers pertinents sont conservés.

.. code-block:: python

    import os 
    dataset_index = 0
    nbr=0
    for chunk_index in range(2600, 54111) :
        audio = fr"C:\Users\ASUS\Desktop\dataset\dataset_{dataset_index}\audio_chunk_{chunk_index}_video_end.wav"
        if os.path.exists(audio) :
            for i in range(chunk_index, chunk_index - 50, -1 ) :
                if os.path.exists(fr"C:\Users\ASUS\Desktop\dataset\dataset_{dataset_index}\audio_chunk_{i}.wav") :
                    os.remove(fr"C:\Users\ASUS\Desktop\dataset\dataset_{dataset_index}\audio_chunk_{i}.wav")
                    os.remove(fr"C:\Users\ASUS\Desktop\dataset\dataset_{dataset_index}\audio_chunk_{i}.txt")
                    nbr +=1
                    print(i)
        if chunk_index%20000 == 0 and chunk_index !=0 :
            dataset_index+=1
    print(f'number of videos that was removed are : {nbr}')

3.2.2 Suppression des longs audios:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ce script supprime automatiquement les fichiers audio dont la durée dépasse 6 secondes, ainsi que leurs fichiers texte associés, à partir d'un dataset organisé en sous-dossiers.

.. code-block:: python
    nbr = 0
    dataset_index = 0
    for chunk_index in range(54111) : ## CHANGE IT TO YOUR MAX CHUNK_INDEX
        try :
            chunk = AudioSegment.from_file(rf'dataset\dataset_{dataset_index}\audio_chunk_{chunk_index}.wav')
        except :
            pass
        if chunk.duration_seconds > 6 :
            if os.path.exists(fr'dataset\dataset_{dataset_index}\audio_chunk_{chunk_index}.wav') :
                os.remove(fr'dataset\dataset_{dataset_index}\audio_chunk_{chunk_index}.wav')
                os.remove(fr'dataset\dataset_{dataset_index}\audio_chunk_{chunk_index}.txt')
                print(F"CHUNK AUDIO {chunk_index} REMOVED")
            nbr +=1
        if chunk_index % 20000 == 0 and chunk_index!=0:
            dataset_index+=1
    print(f"The numbers of audios bigger than 6 seconds are : {nbr}")


3.2.3 Suppression des audios dont leurs transcriptions comportent un mot:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ce script a pour objectif de supprimer les fichiers audio ainsi que leurs fichiers de transcription associés lorsque la transcription ne contient qu’un seul mot (aucun espace dans le texte). Cela permet de nettoyer le dataset en éliminant les segments audio jugés trop courts ou peu pertinents pour des analyses ou traitements ultérieurs.

.. code-block:: python

    def remove_one_word_audios() :
    dataset_index = 0
    nbr = 0
    for chunk_index in range(54111) : ##DONT FORGET TO CHANGE TO YOUR TOTAL CHUNKS
        
        transcription = fr"C:\Users\ASUS\Desktop\dataset\dataset_{dataset_index}\audio_chunk_{chunk_index}.txt"
        audio = fr"C:\Users\ASUS\Desktop\dataset\dataset_{dataset_index}\audio_chunk_{chunk_index}.wav"
        if os.path.exists(transcription) :
            f = open(transcription, "r", encoding="utf-8")
            content = f.read()
            ## REMOVE AUDIOS THAT HAS ONLY ONE SPACE -> ONE WORD
            if content.count(' ') == 0 :
                print(chunk_index)
                nbr+=1
                f.close()
                os.remove(transcription)
                os.remove(audio)
        if chunk_index % 20000 == 0 and chunk_index!=0 :
            dataset_index+=1
    print(f'the number of one word audio are : {nbr}')
 remove_one_word_audios()

3.2.4 Supression des audios de moins de 1 seconde :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ce script vise à supprimer les fichiers audio dont la durée est inférieure à 1 seconde, ainsi que leurs 
fichiers de transcription associés. Cela permet de nettoyer le dataset en éliminant les segments audio très courts, 
souvent inutilisablespour des applications comme l'entraînement de modèles de reconnaissance vocale ou l'analyse audio.

.. code-block:: python

    nbr = 0
    dataset_index = 0
    for chunk_index in range(54111) :
        try :
            chunk = AudioSegment.from_file(fr"C:\Users\ASUS\Desktop\dataset\dataset_{dataset_index}\audio_chunk_{chunk_index}.wav")
        except :
            pass
        if chunk.duration_seconds < 1 :
            if os.path.exists(fr"C:\Users\ASUS\Desktop\dataset\dataset_{dataset_index}\audio_chunk_{chunk_index}.wav") :
                os.remove(fr"C:\Users\ASUS\Desktop\dataset\dataset_{dataset_index}\audio_chunk_{chunk_index}.wav")
                os.remove(fr"C:\Users\ASUS\Desktop\dataset\dataset_{dataset_index}\audio_chunk_{chunk_index}.txt")
                print(f"CHUNK AUDIO {chunk_index} FOUND")
                nbr +=1
        if chunk_index % 20000 == 0 and chunk_index!=0:
            dataset_index+=1
    print(f"The numbers of 0s audios are : {nbr}")

3.2.5 Supression des audios comportant de la musique :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ce script est conçu pour supprimer les fichiers audio et leurs transcriptions associés lorsque la transcription contient 
au moins un caractère [ (généralement utilisé pour indiquer des annotations comme des sons ou de la musique).
L'objectif est de nettoyer le dataset en éliminant les segments correspondant à de la musique, des effets sonores ou 
d'autres annotations non vocales, qui ne sont pas utiles pour des applications de traitement de la parole.

.. code-block:: python
    def remove_one_word_audios() :
    dataset_index = 0
    nbr = 0
    for chunk_index in range(54111) : ##DONT FORGET TO CHANGE TO YOUR TOTAL CHUNKS
        transcription = fr"C:\Users\ASUS\Desktop\dataset\dataset_{dataset_index}\audio_chunk_{chunk_index}.txt" ##REPLACE WITH YOUR DATA PATH
        audio = fr"C:\Users\ASUS\Desktop\dataset\dataset_{dataset_index}\audio_chunk_{chunk_index}.wav"##REPLACE WITH YOUR DATA PATH
        if os.path.exists(transcription) :
            f = open(transcription, "r", encoding="utf-8")
            content = f.read()
            if content.count('[') >=1 :
                print(chunk_index)
                nbr+=1
                f.close()
                os.remove(transcription)
                os.remove(audio)
        if chunk_index % 20000 == 0 and chunk_index!=0 :
            dataset_index+=1           
    print(f'the number of music audios deleted are : {nbr}')
    remove_one_word_audios()

3.2.6 Suppression des audios et de leurs transcriptions contenant des caractères non conformes à la langue arabe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ce script a pour but de supprimer les fichiers audio et leurs transcriptions associés lorsque la transcription 
contient des caractères latins ou des symboles spéciaux qui ne sont pas typiques de la langue arabe. L'objectif 
est d'assurer que le dataset soit exclusivement en arabe et exempt de données qui pourraient perturber les analyses 
ou l'entraînement des modèles linguistiques.

.. code-block:: python
        import os
    def contains_latine(str) :
        latine_special = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        '?', '.', '!', '\\', '-', ';', ':', '"', '“', '%', "'", '�','0',
        '1','2','3','4','5','6','7','8','9',
    ]
        for letter in latine_special :
            if letter in str :
                return True
        return False

    nbr = 0
    dataset_index = 0
    for chunk_index in range(54111) : ##REPLACE WITH YOUR MAX CHUNK_INDEX
        transc_path = fr"C:\Users\ASUS\Desktop\dataset\dataset_{dataset_index}\audio_chunk_{chunk_index}.txt" ##REPLACE WITH YOUR DATA PATH
        audio_path = fr"C:\Users\ASUS\Desktop\dataset\dataset_{dataset_index}\audio_chunk_{chunk_index}.wav" ##REPLACE WITH YOUR DATA PATH
        if os.path.exists(transc_path) :
            transc_file = open(transc_path, 'r', encoding='utf-8')
            transc = transc_file.read()
            transc_file.close()
            if contains_latine(transc) :
                os.remove(transc_path)
                os.remove(audio_path)
                print(f"Chunk {chunk_index} removed !")
                nbr+=1
        if chunk_index % 20000 == 0 and chunk_index!=0 :
                dataset_index+=1
    print(f'The number of latin or special removed audios are {nbr}')