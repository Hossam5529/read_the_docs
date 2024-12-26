V-Modèle traduction 
=======================

Explication du code :
----------------------
Description des Composants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - **AutoTokenizer :**
        - Utilisé pour préparer les données d'entrée du texte (transcription en darija dans ce cas).
        - Il convertit le texte brut en une séquence de tokens compréhensible par le modèle.

    - **AutoModelForSeq2SeqLM :**
        - Charge un modèle pré-entraîné pour la tâche de traduction (Seq2Seq).
        - Dans ce cas, il traduit du darija marocain vers l'anglais.

    - **Modèle pré-entraîné :**
        - Nom du modèle : "centino00/darija-to-english".
        - Ce modèle a été pré-entraîné et adapté pour traduire le texte en darija marocain en anglais.


.. code-block:: python

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("centino00/darija-to-english")
    model = AutoModelForSeq2SeqLM.from_pretrained("centino00/darija-to-english")

Exemple d'Utilisation :
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
    
    # Texte en darija à traduire
    text_darija = "شنو هي الحاله ديالك؟"

    # Tokenisation
    input_ids = tokenizer(text_darija, return_tensors="pt").input_ids

    # Traduction
    output_ids = model.generate(input_ids)

    # Décodage de la traduction
    translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("Traduction en anglais :", translation)


Résultat Attendu :
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pour le texte darija شنو هي الحاله ديالك؟, le modèle devrait fournir une traduction approximative comme :

.. code-block:: python

    Traduction en anglais : What is your condition?

Points Importants :
----------------------

**1-Pré-requis :

    - Le modèle doit être téléchargé depuis Hugging Face. Assurez-vous d'avoir une connexion Internet active lors de l'exécution.
    - Installez transformers si ce n'est pas encore fait :
    .. code-block:: python

        pip install transformers

**2-Adaptabilité :

    - Vous pouvez également utiliser un autre modèle Seq2Seq pour des tâches similaires, comme traduire d'autres langues ou dialectes.

**3-Améliorations Potentielles :

    - Ajouter une gestion des erreurs si le texte fourni dépasse la capacité du modèle ou si le modèle n'est pas trouvé.


