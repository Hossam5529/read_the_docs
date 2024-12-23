Introduction 
================

La reconnaissance automatique de la parole (ASR) et la traduction automatique jouent un rôle clé dans la communication 
et l'accès à l'information, notamment pour les langues et dialectes sous-représentés. Le darija, dialecte arabe parlé au Maroc, 
est un exemple de langue où les ressources numériques et les modèles d'intelligence artificielle sont encore limitées. 
Ce projet vise à développer un système de transcription et traduction de la voix en darija, basé sur les avancées récentes en 
apprentissage profond et en traitement automatique de la langue.

Pour répondre aux défis de cette tâche, nous avons utilisé le modèle Wav2Vec2, un modèle de pointe en reconnaissance vocale, 
conçu pour apprendre efficacement à partir de grandes quantités de données non étiquetées. En le combinant avec un processus 
de fine-tuning, nous avons adapté ce modèle à notre corpus spécifique, collecté à partir de diverses vidéos YouTube contenant 
des échantillons de voix en darija. Cette étape est cruciale pour améliorer les performances du modèle, car elle permet de 
capturer les particularités phonétiques et linguistiques du dialecte marocain.

Notre approche inclut deux étapes principales :

    - **Transcription vocale** : Convertir l'audio en texte en darija, en tirant parti de la puissance du modèle Wav2Vec2.
    - **Traduction automatique** : Traduire le texte transcrit en d'autres langues, telles que le français ou l'anglais, pour élargir son accessibilité.

Ce travail contribue non seulement à enrichir les outils numériques pour le darija, 
mais aussi à renforcer l'inclusion linguistique et culturelle dans les systèmes d'intelligence artificielle.


