# Augmentation Method
# English → French → Spanish → English #

Multilingual translation pipeline to produce paraphrase-like variations of the original text.

## How to Run the Pipeline

# 1. Install Dependencies
# (Python (3.8 or later))
Install the required packages:

pip install -r requirements.txt

# 2. Prepare Dataset
Place any text dataset into the folder: data/original/
The file must be a .txt file and contain one sentence per line

Example:
'The rain in Spain stays mainly in the plain'

# 3. Run the Augmentation Script

From the project directory, execute:
python augment_text.py


The script will:

Read all .txt files in data/original/
Apply multilingual round-trip translation
Generate augmented versions of each sentence

# 4. View Augmented Output

Augmented datasets will be saved automatically to: data/augmented/
Each output file includes both the original and augmented sentences for easy comparison.