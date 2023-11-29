from transformers import BertTokenizer, BertModel, BertConfig
#cd C:\Users\Usuario\mi_entorno_virtual

ruta_destino = r"C:\Users\Usuario\mi_entorno_virtual"

# Definir el modelo BERT en español uncased
modelo_bert_espanol_uncased = "dccuchile/bert-base-spanish-wwm-uncased"

# Cargar el modelo y el tokenizador
modelo = BertModel.from_pretrained(modelo_bert_espanol_uncased)
tokenizador = BertTokenizer.from_pretrained(modelo_bert_espanol_uncased)

# Guardar el modelo en la ruta especificada
modelo.save_pretrained(ruta_destino)
tokenizador.save_pretrained(ruta_destino)

print(f"Modelo BERT en español uncased guardado en: {ruta_destino}")
