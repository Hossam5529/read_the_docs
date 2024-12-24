VI-Application du projet :
===========================

Ce code utilise Streamlit pour créer une interface utilisateur simple permettant de transcrire et de traduire un 
fichier audio parlé en marocain darija. Il intègre les modèles Wav2Vec2 pour la transcription vocale 
et un modèle Seq2Seq pour la traduction en anglais.

6.1 Chargement des modèles et processeurs :
---------------------------------------------

- Les modèles et processeurs sont chargés à l'aide de @st.cache_resource pour éviter de les recharger à chaque interaction. 
- Modèles utilisés :
   - Wav2Vec2 : Pour la transcription vocale en darija.
   - Seq2Seq : Pour la traduction de darija à l'anglais.

.. code-block:: python

    @st.cache_resource
    def load_model():
        processor = Wav2Vec2Processor.from_pretrained("boumehdi/wav2vec2-large-xlsr-moroccan-darija")
        model = Wav2Vec2ForCTC.from_pretrained("boumehdi/wav2vec2-large-xlsr-moroccan-darija")
        return processor, model

    processor, model = load_model()

6.2 Interface utilisateur :
------------------------------
- Titre et description : st.title et st.write affichent des informations sur l'application.
- Chargement de fichiers : st.file_uploader permet à l'utilisateur de télécharger un fichier WAV.

.. code-block:: python

    st.title("Moroccan Darija Speech-to-Text")
    st.write("Upload an audio file to transcribe.")
    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

6.3 Traitement de l'audio :
------------------------------
- L'audio est chargé avec Librosa et converti en un format compatible avec le modèle.
- Les entrées audio sont prétraitées avec le processeur Wav2Vec2.

.. code-block:: python

    if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    st.write("Processing audio...")
    input_audio, sr = librosa.load(uploaded_file, sr=16000)
    input_values = processor(input_audio, return_tensors="pt", padding=True).input_values

6.4 Transcription vocale :
---------------------------

- Les logits (sorties brutes du modèle) sont calculés avec model(input_values).logits.
- Les prédictions de texte sont générées en identifiant les indices les plus probables et en décodant les tokens.

.. code-block:: python.. 

    with torch.no_grad():
       logits = model(input_values).logits
    tokens = torch.argmax(logits, axis=-1)
    transcription = processor.batch_decode(tokens, skip_special_tokens=True)
    st.subheader("Transcription:")
    st.write(transcription[0])


6.5 Traduction de la transcription :
-------------------------------------

- Un modèle Seq2Seq est utilisé pour traduire la transcription en anglais.
- Les données sont tokenisées, passées dans le modèle, et décodées.

 .. code-block:: python

    tokenizer1 = AutoTokenizer.from_pretrained("centino00/darija-to-english")
    model1 = AutoModelForSeq2SeqLM.from_pretrained("centino00/darija-to-english")
    input_ids = tokenizer1(transcription[0], return_tensors="pt").input_ids 
    generated_ids = model1.generate(input_ids)
    output = tokenizer1.decode(generated_ids[0], skip_special_tokens=True)
    st.subheader("Translation:")
    st.write(output)




       




