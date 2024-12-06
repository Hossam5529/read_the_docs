import yt_dlp as youtube_dl  # Now using yt-dlp for better support
import re
import os
from pydub import AudioSegment
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi, VideoUnavailable, NoTranscriptFound, TranscriptsDisabled
#from get_urls import url_list


url_list = ['https://www.youtube.com/watch?v=0HADc9HYHEo', 'https://www.youtube.com/watch?v=ydPBzfVX19w', 'https://www.youtube.com/watch?v=8vki4l-WTmw', 'https://www.youtube.com/watch?v=L2rzIt7cipc', 'https://www.youtube.com/watch?v=jo6FlMhlUiY', 'https://www.youtube.com/watch?v=0HADc9HYHEo', 'https://www.youtube.com/watch?v=ydPBzfVX19w', 'https://www.youtube.com/watch?v=8vki4l-WTmw', 'https://www.youtube.com/watch?v=L2rzIt7cipc', 'https://www.youtube.com/watch?v=jo6FlMhlUiY', 'https://www.youtube.com/watch?v=Mcr_es0e5zo', 'https://www.youtube.com/watch?v=QZngx5Evuf8', 'https://www.youtube.com/watch?v=Tbtyj64Cec0', 'https://www.youtube.com/watch?v=Q3KzqFeqoIA', 'https://www.youtube.com/watch?v=d1fW65obBEw', 'https://www.youtube.com/watch?v=AyvaLH7N6QE', 'https://www.youtube.com/watch?v=scg7vBwQzqI', 'https://www.youtube.com/watch?v=7daJZfKF8bM', 'https://www.youtube.com/watch?v=d1fW65obBEw', 'https://www.youtube.com/watch?v=AyvaLH7N6QE', 'https://www.youtube.com/watch?v=scg7vBwQzqI', 'https://www.youtube.com/watch?v=7daJZfKF8bM', 'https://www.youtube.com/watch?v=AyvaLH7N6QE', 'https://www.youtube.com/watch?v=scg7vBwQzqI', 'https://www.youtube.com/watch?v=7daJZfKF8bM', 'https://www.youtube.com/watch?v=rzm_PTBIJYY', 'https://www.youtube.com/watch?v=e9nV1XaqZlU', 'https://www.youtube.com/watch?v=DaHMiUOc_xQ', 'https://www.youtube.com/watch?v=uccnpCUO68g', 'https://www.youtube.com/watch?v=tt-_mEh6C5s', 'https://www.youtube.com/watch?v=scg7vBwQzqI', 'https://www.youtube.com/watch?v=7daJZfKF8bM', 'https://www.youtube.com/watch?v=rzm_PTBIJYY', 'https://www.youtube.com/watch?v=e9nV1XaqZlU', 'https://www.youtube.com/watch?v=DaHMiUOc_xQ', 'https://www.youtube.com/watch?v=uccnpCUO68g', 'https://www.youtube.com/watch?v=tt-_mEh6C5s', 'https://www.youtube.com/watch?v=SAF7OzzzQbY', 'https://www.youtube.com/watch?v=gh0fKMtcgmo', 'https://www.youtube.com/watch?v=7ctSmP6oEok', 'https://www.youtube.com/watch?v=WzQvllrKTtg', 'https://www.youtube.com/watch?v=rzm_PTBIJYY', 'https://www.youtube.com/watch?v=e9nV1XaqZlU', 'https://www.youtube.com/watch?v=DaHMiUOc_xQ', 'https://www.youtube.com/watch?v=uccnpCUO68g', 'https://www.youtube.com/watch?v=tt-_mEh6C5s', 'https://www.youtube.com/watch?v=SAF7OzzzQbY', 'https://www.youtube.com/watch?v=gh0fKMtcgmo', 'https://www.youtube.com/watch?v=7ctSmP6oEok', 'https://www.youtube.com/watch?v=WzQvllrKTtg', 'https://www.youtube.com/watch?v=1ZMF034C8Bg', 'https://www.youtube.com/watch?v=uccnpCUO68g', 'https://www.youtube.com/watch?v=tt-_mEh6C5s', 'https://www.youtube.com/watch?v=SAF7OzzzQbY', 'https://www.youtube.com/watch?v=gh0fKMtcgmo', 'https://www.youtube.com/watch?v=7ctSmP6oEok', 'https://www.youtube.com/watch?v=WzQvllrKTtg', 'https://www.youtube.com/watch?v=1ZMF034C8Bg', 'https://www.youtube.com/watch?v=SAF7OzzzQbY', 'https://www.youtube.com/watch?v=gh0fKMtcgmo', 'https://www.youtube.com/watch?v=7ctSmP6oEok', 'https://www.youtube.com/watch?v=WzQvllrKTtg', 'https://www.youtube.com/watch?v=1ZMF034C8Bg', 'https://www.youtube.com/watch?v=1-ObREw1LYk', 'https://www.youtube.com/watch?v=Dk37DEeFqHM', 'https://www.youtube.com/watch?v=XdK7vE_1R68', 'https://www.youtube.com/watch?v=1ZMF034C8Bg', 'https://www.youtube.com/watch?v=1-ObREw1LYk', 'https://www.youtube.com/watch?v=Dk37DEeFqHM', 'https://www.youtube.com/watch?v=XdK7vE_1R68', 'https://www.youtube.com/watch?v=1-ObREw1LYk', 'https://www.youtube.com/watch?v=Dk37DEeFqHM', 'https://www.youtube.com/watch?v=XdK7vE_1R68', 'https://www.youtube.com/watch?v=neySDXX9WAo', 'https://www.youtube.com/watch?v=XdK7vE_1R68', 'https://www.youtube.com/watch?v=neySDXX9WAo', 'https://www.youtube.com/watch?v=neySDXX9WAo', 'https://www.youtube.com/watch?v=QZ_NVRkwx1I']




def download_audio(yt_url, folder_name, audio_name):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{folder_name}/{audio_name}',  # Save without .mp3 extension
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([yt_url])
        print(f"Audio downloaded for {yt_url} and saved as {audio_name} in {folder_name}")
    except Exception as e:
        print(f"An error occurred while downloading audio: {e}")
        
def convert_to_wav(folder_name, audio_name):
    input_file = f"{folder_name}/{audio_name}.mp3"
    output_file = f"{folder_name}/{audio_name}.wav"
    
    # Load and convert the audio file
    audio = AudioSegment.from_mp3(input_file)
    audio.export(output_file, format="wav")
    
    # Remove the original .mp3 file
    if os.path.exists(output_file):
        os.remove(input_file)
        print(f"Conversion complete! '{input_file}' has been replaced by '{output_file}'.")
    else:
        print("Conversion failed; .wav file was not created.")
        


def cut_audio(input_file, output_file, start_time, end_time):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Cut the audio
    cut_audio = audio[start_time:end_time]

    # Save the cut audio to a new file
    cut_audio.export(output_file, format = 'wav')

def get_starts(video_id) :
    transcription = YouTubeTranscriptApi.get_transcript(video_id,  languages=['ar'])
    starts = []
    for element in transcription :
        starts.append(element['start'])
    return starts

def get_ends(video_id) :
    transcription = YouTubeTranscriptApi.get_transcript(video_id,  languages=['ar'])
    ends = []
    for element in transcription :
        ends.append(element['start'] + element['duration'])
    return ends

def get_transcriptions(video_id) :
    transcription = YouTubeTranscriptApi.get_transcript(video_id,  languages=['ar'])
    transcriptions = []
    for element in transcription :
        transcriptions.append(element['text'])
    return transcriptions



def get_video_id(url) :
    id = url.rsplit("=")
    return id[1]



def process_videos(url_list):
    global_chunk_index = 0
    current_folder_index =21
    folder_name = f"dataset{current_folder_index}"

    # Créer le dossier initial
    os.makedirs(folder_name, exist_ok=True)

    for i, url in enumerate(url_list, start=1):
        try:
            audio_name = f"audio_{i}"
            audio_file = os.path.join(folder_name, f'audio_{i}.mp3')
            transcription_name = f"transcription_{i}"

            # Étape 1 : Télécharger l'audio
            try:
                download_audio(url, folder_name, audio_name)
            except Exception as e:
                print(f"Erreur lors du téléchargement de l'audio pour {url}: {e}")
                continue

            # Étape 2 : Vérifier si le fichier audio a été téléchargé
            if not os.path.exists(audio_file):
                print(f"Fichier audio introuvable pour {url}. Passage au suivant.")
                continue

            # Étape 3 : Changer la fréquence d'échantillonnage
            try:
                audio_mp3 = AudioSegment.from_file(audio_file)
                resampled_audio = audio_mp3.set_frame_rate(16000)
                resampled_audio.export(audio_file, format="mp3")
            except Exception as e:
                print(f"Erreur lors du traitement du fichier MP3 {audio_file}: {e}")
                continue

            # Étape 4 : Convertir en WAV
            try:
                convert_to_wav(folder_name, audio_name)
            except Exception as e:
                print(f"Erreur lors de la conversion en WAV pour {audio_file}: {e}")
                continue

            input_file = os.path.join(folder_name, f'audio_{i}.wav')
            if not os.path.exists(input_file):
                print(f"Fichier WAV introuvable après conversion pour {url}. Passage au suivant.")
                continue

            # Étape 5 : Récupérer les métadonnées (transcriptions, débuts, fins)
            try:
                video_id = get_video_id(url)
                transcriptions = get_transcriptions(video_id)
                starts = get_starts(video_id)
                ends = get_ends(video_id)

                if not (transcriptions and starts and ends):
                    print(f"Métadonnées manquantes pour la vidéo {url}. Passage au suivant.")
                    continue
            except Exception as e:
                print(f"Erreur lors de l'extraction des métadonnées pour {url}: {e}")
                continue

            # Étape 6 : Découper l'audio en chunks
            for j in range(len(starts)):
                try:
                    if global_chunk_index >= 10000:
                        global_chunk_index = 0
                        current_folder_index += 1
                        folder_name = f"dataset{current_folder_index}"
                        os.makedirs(folder_name, exist_ok=True)

                    output_file = os.path.join(folder_name, f'audio_chunk_{global_chunk_index}.wav')
                    transcription_file = os.path.join(folder_name, f'audio_chunk_{global_chunk_index}.txt')

                    start_time = starts[j] * 1000 - 150
                    if start_time < 0:
                        start_time += 150
                    if j + 1 < len(starts):
                        end_time = starts[j + 1] * 1000 + 150
                    else:
                        end_time = ends[j] * 1000

                    # Découper l'audio
                    cut_audio(input_file, output_file, start_time, end_time)

                    # Écrire la transcription
                    with open(transcription_file, "w", encoding="utf-8") as f:
                        f.write(transcriptions[j])

                    global_chunk_index += 1
                except Exception as e:
                    print(f"Erreur lors du traitement du chunk {global_chunk_index} pour {url}: {e}")
                    continue

            # Étape 7 : Supprimer le fichier WAV temporaire
            try:
                os.remove(input_file)
            except OSError as e:
                print(f"Erreur lors de la suppression du fichier temporaire {input_file}: {e}")
        except Exception as e:
            print(f"Erreur inattendue pour {url}: {e}")




if __name__ == "__main__":
    process_videos(url_list)