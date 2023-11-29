from transformers import BertModel, BertTokenizer

# Especifica la ruta del directorio de destino

ruta_destino = r"C:\Users\Usuario\mi_entorno_virtual\RASA_PRYECT"

# Especifica el modelo BERT en español uncased
modelo_bert_espanol_uncased = "dccuchile/bert-base-spanish-wwm-uncased"

# Cargar el modelo y el tokenizador
modelo = BertModel.from_pretrained(modelo_bert_espanol_uncased)
tokenizador = BertTokenizer.from_pretrained(modelo_bert_espanol_uncased)

# Guardar el modelo y el tokenizador en el directorio especificado
modelo.save_pretrained(ruta_destino)
tokenizador.save_pretrained(ruta_destino)

print(f"Modelo BERT en español uncased y tokenizador guardados en: {ruta_destino}")


