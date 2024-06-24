# Importuje klasę models z modułu django.db
from django.db import models
# Importuje moduł numpy i przypisuje mu alias np
from django.core.files.base import ContentFile  # Importuje klasę ContentFile z django.core.files.base
import numpy as np
# Importuje default_storage z modułu django.core.files.storage
from django.core.files.storage import default_storage
# Importuje moduł image z tensorflow.keras.preprocessing i przypisuje mu alias tf_image
from tensorflow.keras.preprocessing import image as tf_image
# Importuje klasy i funkcje z inception_v3
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
#Importowanie modułu openai
from openai import OpenAI
import requests
api_key="sk-proj-BG7v7vDbWFBuxGLa6ZqST3BlbkFJvycOEC7a2kY55UhdIOSp"
client = OpenAI(api_key=api_key)

#Model
class AIResponse(models.Model):
    title = models.CharField(max_length=255, blank=True, null=True)
    content = models.TextField(blank=True, null=True)
    request = models.CharField(max_length=255, blank=True)
    response = models.ImageField(upload_to="mediaphoto", blank=True, null=True)
    

    def __str__(self):
        return self.request

    def save(self, *args, **kwargs):
        if not self.response:  # Sprawdzanie, czy pole `response` jest puste
            # Generowanie obrazu za pomocą OpenAI
            response = client.images.generate(
                model="dall-e-2",
                prompt=self.request,
                size="1024x1024",
                n=1
            )

            # Pobieranie URL wygenerowanego obrazu
            image_url = response.data[0].url

            # Pobieranie obrazu z URL
            image_response = requests.get(image_url)
            if image_response.status_code == 200:
                image_name = f"{self.request.replace(' ', '_')}.png"  # Nazwa pliku
                self.response.save(image_name, ContentFile(image_response.content), save=False)
        
        super().save(*args, **kwargs)

        if self.response:
            try:
                file_path = self.response.path
                if default_storage.exists(file_path):
                    # Ładowanie obrazu o zadanych wymiarach
                    pil_image = tf_image.load_img(file_path, target_size=(299, 299))
                    # Konwersja obrazu na tablicę numpy
                    img_array = tf_image.img_to_array(pil_image)
                    # Rozszerzanie tablicy o nowy wymiar
                    img_array = np.expand_dims(img_array, axis=0)
                    # Przetwarzanie obrazu zgodnie z wymaganiami modelu
                    img_array = preprocess_input(img_array)

                    # Tworzenie modelu InceptionV3
                    model = InceptionV3(weights='imagenet')
                    # Dokonywanie predykcji na obrazie
                    predictions = model.predict(img_array)
                    # Dekodowanie predykcji
                    decoded_predictions = decode_predictions(predictions, top=1)[0]
                    # Wybieranie najbardziej prawdopodobnej etykiety
                    best_guess = decoded_predictions[0][1]
                    # Ustawianie tytułu na najbardziej prawdopodobną etykietę
                    self.title = best_guess
                    # Tworzenie łańcucha znaków zawierającego etykiety i prawdopodobieństwa predykcji
                    self.content = ', '.join([f"{pred[1]}: {pred[2] * 100:.2f}%" for pred in decoded_predictions])
                    super().save(*args, **kwargs)

            except Exception as e:
                pass