from transformers import BertModel, BertTokenizer

# Especifica el modelo y el tokenizador
model_name = "dccuchile/bert-base-spanish-wwm-uncased"
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Especifica la carpeta donde deseas guardar el modelo
folder_path = r"C:\Users\Usuario\Desktop\chatbot\RASA_PRYECT\models"

# Guarda el modelo y el tokenizador en la carpeta especificada
model.save_pretrained(folder_path)
tokenizer.save_pretrained(folder_path)

print(f"Modelo y tokenizador guardados en: {folder_path}")

#C:/Users/Usuario/Scripts/activate  # Activar el entorno virtual (Windows)
#python r"C:\Users\Usuario\Desktop\chatbot\RASA_PRYECT\models\model_1.py"
