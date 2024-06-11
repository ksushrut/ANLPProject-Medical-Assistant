import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class_id_to_disease_info = {
    0: {'name': '(vertigo) paroymsal positional vertigo', 'precautions': ['lie down, avoid sudden change in body, avoid abrupt head movement, relax']},
    1: {'name': 'acne', 'precautions': ['bath twice, avoid fatty spicy food, drink plenty of water, avoid too many products']},
    2: {'name': 'aids', 'precautions': ['avoid open cuts, wear ppe if possible, consult doctor, follow up']},
    3: {'name': 'alcoholic hepatitis', 'precautions': ['stop alcohol consumption, consult doctor, medication, follow up']},
    4: {'name': 'arthritis', 'precautions': ['exercise, use hot and cold therapy, try acupuncture, massage']},
    5: {'name': 'bronchial asthma', 'precautions': ['switch to loose clothing, take deep breaths, get away from trigger, seek help']},
    6: {'name': 'cervical spondylosis', 'precautions': ['use heating pad or cold pack, exercise, take otc pain reliever, consult doctor']},
    7: {'name': 'chicken pox', 'precautions': ['use neem in bathing, consume neem leaves, take vaccine, avoid public places']},
    8: {'name': 'chronic cholestasis', 'precautions': ['cold baths, anti itch medicine, consult doctor, eat healthy']},
    9: {'name': 'common cold', 'precautions': ['drink vitamin c rich drinks, take vapor, avoid cold food, keep fever in check']},
    10: {'name': 'dengue', 'precautions': ['drink papaya leaf juice, avoid fatty spicy food, keep mosquitoes away, keep hydrated']},
    11: {'name': 'diabetes', 'precautions': ['have balanced diet, exercise, consult doctor, follow up']},
    12: {'name': 'dimorphic hemmorhoids(piles)', 'precautions': ['avoid fatty spicy food, consume witch hazel, warm bath with epsom salt, consume aloe vera juice']},
    13: {'name': 'fungal infection', 'precautions': ['bath twice, use detol or neem in bathing water, keep infected area dry, use clean cloths']},
    14: {'name': 'gastroenteritis', 'precautions': ['stop eating solid food for while, try taking small sips of water, rest, ease back into eating']},
    15: {'name': 'gerd', 'precautions': ['avoid fatty spicy food, avoid lying down after eating, maintain healthy weight, exercise']},
    16: {'name': 'hepatitis a', 'precautions': ['Consult nearest hospital, wash hands thoroughly, avoid fatty spicy food, medication']},
    17: {'name': 'hepatitis b', 'precautions': ['consult nearest hospital, vaccination, eat healthy, medication']},
    18: {'name': 'hepatitis c', 'precautions': ['Consult nearest hospital, vaccination, eat healthy, medication']},
    19: {'name': 'hepatitis d', 'precautions': ['consult doctor, medication, eat healthy, follow up']},
    20: {'name': 'hepatitis e', 'precautions': ['stop alcohol consumption, rest, consult doctor, medication']},
    21: {'name': 'hypertension', 'precautions': ['meditation, salt baths, reduce stress, get proper sleep']},
    22: {'name': 'hyperthyroidism', 'precautions': ['eat healthy, massage, use lemon balm, take radioactive iodine treatment']},
    23: {'name': 'hypoglycemia', 'precautions': ['lie down on side, check in pulse, drink sugary drinks, consult doctor']},
    24: {'name': 'hypothyroidism', 'precautions': ['reduce stress, exercise, eat healthy, get proper sleep']},
    25: {'name': 'impetigo', 'precautions': ['soak affected area in warm water, use antibiotics, remove scabs with wet compressed cloth, consult doctor']},
    26: {'name': 'jaundice', 'precautions': ['drink plenty of water, consume milk thistle, eat fruits and high fibrous food, medication']},
    27: {'name': 'malaria', 'precautions': ['Consult nearest hospital, avoid oily food, avoid non veg food, keep mosquitoes out']},
    28: {'name': 'migraine', 'precautions': ['meditation, reduce stress, use polaroid glasses in sun, consult doctor']},
    29: {'name': 'osteoarthristis', 'precautions': ['acetaminophen, consult nearest hospital, follow up, salt baths']},
    30: {'name': 'paralysis (brain hemorrhage)', 'precautions': ['massage, eat healthy, exercise, consult doctor']},
    31: {'name': 'peptic ulcer disease', 'precautions': ['avoid fatty spicy food, consume probiotic food, eliminate milk, limit alcohol']},
    32: {'name': 'pneumonia', 'precautions': ['consult doctor, medication, rest, follow up']},
    33: {'name': 'psoriasis', 'precautions': ['wash hands with warm soapy water, stop bleeding using pressure, consult doctor, salt baths']},
    34: {'name': 'tuberculosis', 'precautions': ['cover mouth, consult doctor, medication, rest']},
    35: {'name': 'typhoid', 'precautions': ['eat high calorie vegetables, antibiotic therapy, consult doctor, medication']},
    36: {'name': 'urinary tract infection', 'precautions': ['drink plenty of water, increase vitamin c intake, drink cranberry juice, take probiotics']},
    37: {'name': 'varicose veins', 'precautions': ['lie down flat and raise the leg high, use ointments, use vein compression, don\'t stand still for long']}
}


def load_model(model_checkpoint="roberta-base", model_path="roberta"):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=38)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return model, tokenizer

def predict(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    disease_info = class_id_to_disease_info.get(predicted_class, {"name": "Unknown Disease", "precautions": []})
    return disease_info["name"], disease_info["precautions"]

