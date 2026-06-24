import math
import random
import re

# Set random seed for reproducibility
random.seed(42)

# 1. Generate synthetic clinical notes representing typical variations, synonyms, and abbreviations
symptoms = [
    # Headache variations
    ["cefaleia frontal", "dor de cabeca", "dor de cabeça", "enxaqueca", "cefaleia holocraniana", "throbbing headache", "migraine"],
    # Diabetes variations
    ["DM2", "DM", "diabetes mellitus", "diabetes tipo 2", "glicemia alta", "hyperglycemia", "diabetes"],
    # Hypertension variations
    ["HAS", "hipertensao arterial", "pressao alta", "pressão alta", "hipertensão", "HTN", "hypertension"],
    # Fever variations
    ["febre", "estado febril", "pirexia", "temperatura elevada", "fever", "pyrexia"],
    # Dyspnea variations
    ["dispneia", "falta de ar", "cansaco respiratorio", "cansaço respiratório", "shortness of breath", "dyspnea"]
]

medications = [
    ["metformina", "Glifage", "metformin"],
    ["losartana", "Aradois", "losartan"],
    ["paracetamol", "Tylenol", "acetaminophen"],
    ["ibuprofeno", "Advil", "ibuprofen"],
    ["insulina", "Lantus", "insulin"]
]

templates = [
    "Paciente queixa-se de {symptom1} há {dur}. Refere diagnóstico de {symptom2} em uso de {med}.",
    "Quadro de {symptom1} associado a {symptom2}. Faz uso regular de {med}.",
    "Admitido com {symptom1} intensa. Nega outras queixas. Histórico de {symptom2}. Uso de {med}.",
    "Apresenta {symptom1}. Relata {symptom2} controlado com {med}.",
    "Evolução clínica: {symptom1} resolvida, mas mantém {symptom2}. Ajustado {med}."
]

durations = ["3 dias", "1 semana", "5 dias", "meses", "24h"]

def generate_mock_notes(count=150):
    notes = []
    for _ in range(count):
        s1 = random.choice(random.choice(symptoms))
        s2 = random.choice(random.choice(symptoms))
        while s2 == s1:
            s2 = random.choice(random.choice(symptoms))
        med = random.choice(random.choice(medications))
        dur = random.choice(durations)
        
        template = random.choice(templates)
        note = template.format(symptom1=s1, symptom2=s2, med=med, dur=dur)
        notes.append(note)
    return notes

# 2. Tokenizer to extract medical unigrams and bigrams
def extract_terms(text):
    # Lowercase and clean text
    clean_text = re.sub(r'[^\w\s-]', '', text.lower())
    words = clean_text.split()
    
    # We want to filter out common Portuguese/English stop words to focus on clinical terms
    stopwords = {
        'o', 'a', 'os', 'as', 'de', 'do', 'da', 'dos', 'das', 'em', 'um', 'uma', 
        'com', 'para', 'por', 'que', 'se', 'há', 'faz', 'em', 'uso', 'histórico', 
        'relata', 'nega', 'outras', 'queixas', 'admitido', 'apresenta', 'paciente',
        'queixa-se', 'regime', 'regular', 'controle', 'controlado', 'em', 'uso',
        'and', 'the', 'of', 'in', 'with', 'for', 'to', 'on', 'at', 'by', 'an', 
        'presents', 'complains', 'history', 'diagnosis'
    }
    
    unigrams = [w for w in words if w not in stopwords and len(w) > 2]
    
    # Generate bigrams
    bigrams = []
    for i in range(len(words) - 1):
        if words[i] not in stopwords and words[i+1] not in stopwords:
            bigrams.append(f"{words[i]} {words[i+1]}")
            
    return unigrams + bigrams

def main():
    print("=== Generating Synthetic Clinical Corpus ===")
    corpus = generate_mock_notes(150)
    print(f"Generated {len(corpus)} medical records representing abbreviation and synonym variants.\n")
    
    # 3. Simulate annotation process note-by-line
    accumulated_vocabulary = set()
    vocabulary_growth = []
    notes_processed = []
    
    for idx, note in enumerate(corpus):
        terms = extract_terms(note)
        accumulated_vocabulary.update(terms)
        
        notes_processed.append(idx + 1)
        vocabulary_growth.append(len(accumulated_vocabulary))
        
    print("=== Vocabulary Growth Simulation ===")
    print(f"Total Unique Vocabulary Terms Discovered: {vocabulary_growth[-1]}")
    
    # 4. Calculate Saturation Threshold
    # We define saturation as the point where labeling an additional batch of 10 notes 
    # yields less than a 2% increase in new unique clinical terms.
    batch_size = 10
    saturation_point = None
    
    print("\nBatch Analysis (New Term Discovery Rate):")
    print(f"{'Batch Range':<15} | {'Unique Vocab Size':<18} | {'New Terms Found':<16} | {'Growth Rate (%)':<15}")
    print("-" * 72)
    
    for i in range(batch_size, len(corpus) + 1, batch_size):
        vocab_current = vocabulary_growth[i - 1]
        vocab_prev = vocabulary_growth[i - batch_size - 1] if i > batch_size else 0
        new_terms = vocab_current - vocab_prev
        growth_rate = (new_terms / vocab_prev * 100) if vocab_prev > 0 else 100
        
        print(f"Notes {i-batch_size+1:>3}-{i:<3}   | {vocab_current:<18} | {new_terms:<16} | {growth_rate:.2f}%")
        
        # Check for saturation (< 3% discovery rate)
        if growth_rate < 3.0 and saturation_point is None and i > 20:
            saturation_point = i
            
    print("-" * 72)
    
    if saturation_point:
        print(f"\n🎯 Recommended Labeled Sample Size: {saturation_point} notes.")
        print(f"Reason: Beyond {saturation_point} annotated notes, the new term discovery rate drops below 3%.")
        print("At this stage, your models will have observed ~90% of local terminology variants.")
    else:
        print("\n⚠️ No saturation point reached within 150 notes. Recommend increasing corpus size.")

    # 5. Output ASCII visualization of the saturation curve
    print("\n=== Saturation Curve Preview (ASCII) ===")
    max_vocab = vocabulary_growth[-1]
    steps = 15
    step_size = len(corpus) // steps
    
    for i in range(0, len(corpus), step_size):
        vocab_size = vocabulary_growth[i]
        stars = int((vocab_size / max_vocab) * 30)
        print(f"Notes Labeled: {i+1:>3} | " + "*" * stars + " " * (30 - stars) + f" ({vocab_size} terms)")

if __name__ == "__main__":
    main()
