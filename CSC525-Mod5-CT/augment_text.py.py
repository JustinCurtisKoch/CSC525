import os
from transformers import pipeline

# -------------------------------
# Configuration
# -------------------------------
INPUT_DIR = "data/original"
OUTPUT_DIR = "data/augmented"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Load Translation Pipelines
# -------------------------------
en_to_fr = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
fr_to_es = pipeline("translation_fr_to_es", model="Helsinki-NLP/opus-mt-fr-es")
es_to_en = pipeline("translation_es_to_en", model="Helsinki-NLP/opus-mt-es-en")

# -------------------------------
# Round-Trip Translation Function
# -------------------------------
def round_trip_translate(text):
    french = en_to_fr(text)[0]["translation_text"]
    spanish = fr_to_es(french)[0]["translation_text"]
    english = es_to_en(spanish)[0]["translation_text"]
    return english

# -------------------------------
# Augment All Text Files
# -------------------------------
for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".txt"):
        continue

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(
        OUTPUT_DIR, filename.replace(".txt", "_augmented.txt")
    )

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            augmented_text = round_trip_translate(line)

            outfile.write(f"ORIGINAL: {line}\n")
            outfile.write(f"AUGMENTED: {augmented_text}\n\n")

    print(f"Augmented dataset saved to: {output_path}")
