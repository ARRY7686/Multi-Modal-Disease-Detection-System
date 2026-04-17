"""
Synthetic symptom text generator.

Creates realistic clinical symptom descriptions for each of the 25 disease
labels.  Supports:
  - Template-based generation with randomised demographics / severity
  - GPT augmentation prompt generation for richer diversity
  - Multi-label: generating descriptions for co-occurring diseases

Usage (standalone):
    python utils/synthetic_symptoms.py --output data/symptoms.csv --per-disease 200
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Disease → symptom templates
# ──────────────────────────────────────────────────────────────────────────────

DISEASE_SYMPTOM_MAP: Dict[str, Dict] = {
    # ==================== CHEST (0-13) ====================
    "Atelectasis": {
        "symptoms": [
            "reduced breath sounds on one side",
            "mild dyspnea",
            "low-grade fever",
            "cough",
            "tachypnea",
            "chest discomfort",
        ],
        "contexts": [
            "post-operative patient",
            "prolonged bed rest",
            "recent intubation",
        ],
        "severity_range": (2, 6),
    },
    "Cardiomegaly": {
        "symptoms": [
            "progressive dyspnea on exertion",
            "orthopnea",
            "bilateral lower extremity edema",
            "fatigue",
            "paroxysmal nocturnal dyspnea",
            "palpitations",
            "jugular venous distention",
        ],
        "contexts": [
            "known history of hypertension",
            "prior myocardial infarction",
            "valvular heart disease",
            "chronic alcohol use",
        ],
        "severity_range": (4, 9),
    },
    "Effusion": {
        "symptoms": [
            "dyspnea that worsens when lying flat",
            "pleuritic chest pain",
            "reduced breath sounds at the base",
            "dullness to percussion",
            "dry cough",
        ],
        "contexts": [
            "known malignancy",
            "recent pneumonia",
            "heart failure history",
            "liver cirrhosis",
        ],
        "severity_range": (3, 7),
    },
    "Infiltration": {
        "symptoms": [
            "productive cough",
            "mild fever",
            "fatigue",
            "decreased appetite",
            "mild chest discomfort",
        ],
        "contexts": [
            "community-acquired infection",
            "immunocompromised state",
        ],
        "severity_range": (2, 5),
    },
    "Mass": {
        "symptoms": [
            "persistent cough lasting more than 3 weeks",
            "unintentional weight loss",
            "hemoptysis",
            "chest pain",
            "recurrent pneumonia in the same lobe",
        ],
        "contexts": [
            "long history of smoking",
            "occupational asbestos exposure",
            "family history of lung cancer",
        ],
        "severity_range": (4, 8),
    },
    "Nodule": {
        "symptoms": [
            "incidental finding on chest X-ray",
            "no specific symptoms",
            "mild persistent cough",
        ],
        "contexts": [
            "routine health screening",
            "smoker over 50 years old",
            "history of granulomatous disease",
        ],
        "severity_range": (1, 3),
    },
    "Pneumonia": {
        "symptoms": [
            "high fever with chills and rigors",
            "productive cough with purulent sputum",
            "severe dyspnea",
            "pleuritic chest pain",
            "tachycardia",
            "malaise and body aches",
        ],
        "contexts": [
            "recent upper respiratory infection",
            "elderly patient in nursing home",
            "immunosuppressed individual",
        ],
        "severity_range": (5, 9),
    },
    "Pneumothorax": {
        "symptoms": [
            "sudden onset sharp chest pain",
            "acute dyspnea",
            "reduced breath sounds on affected side",
            "hypoxia",
            "tachycardia",
        ],
        "contexts": [
            "tall thin young male",
            "recent thoracic procedure",
            "underlying COPD",
            "mechanical ventilation",
        ],
        "severity_range": (6, 10),
    },
    "Consolidation": {
        "symptoms": [
            "productive cough",
            "fever",
            "bronchial breath sounds",
            "increased tactile fremitus",
            "malaise",
        ],
        "contexts": [
            "community-acquired pneumonia",
            "aspiration event",
        ],
        "severity_range": (4, 7),
    },
    "Edema": {
        "symptoms": [
            "severe dyspnea at rest",
            "orthopnea requiring multiple pillows",
            "pink frothy sputum",
            "bilateral crackles on auscultation",
            "anxiety and restlessness",
        ],
        "contexts": [
            "acute heart failure exacerbation",
            "fluid overload",
            "renal failure",
        ],
        "severity_range": (7, 10),
    },
    "Emphysema": {
        "symptoms": [
            "chronic progressive dyspnea",
            "pursed lip breathing",
            "barrel chest appearance",
            "prolonged expiratory phase",
            "decreased breath sounds diffusely",
            "exercise intolerance",
        ],
        "contexts": [
            "long-term smoker",
            "alpha-1 antitrypsin deficiency",
        ],
        "severity_range": (4, 8),
    },
    "Fibrosis": {
        "symptoms": [
            "progressive exertional dyspnea",
            "dry non-productive cough",
            "fine bibasilar crackles (velcro-like)",
            "digital clubbing",
            "fatigue",
        ],
        "contexts": [
            "idiopathic pulmonary fibrosis",
            "occupational dust exposure",
            "connective tissue disease",
        ],
        "severity_range": (4, 8),
    },
    "Pleural_Thickening": {
        "symptoms": [
            "mild chronic chest pain",
            "decreased breath sounds at the base",
            "mild dyspnea on exertion",
        ],
        "contexts": [
            "asbestos exposure history",
            "previous pleural infection",
            "prior hemothorax",
        ],
        "severity_range": (2, 5),
    },
    "Hernia": {
        "symptoms": [
            "epigastric fullness after meals",
            "shortness of breath when supine",
            "bowel sounds heard in the chest",
            "nausea and vomiting",
        ],
        "contexts": [
            "previous abdominal surgery",
            "congenital diaphragmatic defect",
            "traumatic injury to the diaphragm",
        ],
        "severity_range": (3, 7),
    },
    # ==================== SKIN (14-20) ====================
    "Melanoma": {
        "symptoms": [
            "asymmetric mole with irregular borders",
            "mole with multiple colors (brown, black, red, white, blue)",
            "diameter greater than 6 mm",
            "recent change in size, shape, or color",
            "itching or bleeding from a mole",
            "new pigmented lesion that looks different from others",
        ],
        "contexts": [
            "fair-skinned individual with history of sunburns",
            "family history of melanoma",
            "large number of atypical nevi",
            "prior melanoma diagnosis",
        ],
        "severity_range": (6, 10),
    },
    "Melanocytic_Nevi": {
        "symptoms": [
            "small round symmetric mole",
            "uniform brown color",
            "well-defined borders",
            "diameter less than 6 mm",
            "stable appearance over time",
        ],
        "contexts": [
            "routine skin examination",
            "cosmetic concern",
        ],
        "severity_range": (1, 2),
    },
    "Basal_Cell_Carcinoma": {
        "symptoms": [
            "pearly or waxy bump on the skin",
            "flat flesh-colored or brown scar-like lesion",
            "bleeding or scabbing sore that heals and returns",
            "visible telangiectatic vessels on the surface",
        ],
        "contexts": [
            "sun-exposed area (face, neck, ears)",
            "chronic sun exposure history",
            "fair skin type",
        ],
        "severity_range": (3, 6),
    },
    "Actinic_Keratosis": {
        "symptoms": [
            "rough scaly patch on the skin",
            "pink or red raised area",
            "tenderness when touched",
            "may itch intermittently",
        ],
        "contexts": [
            "chronic sun exposure",
            "age over 50",
            "light skin complexion",
        ],
        "severity_range": (2, 5),
    },
    "Benign_Keratosis": {
        "symptoms": [
            "waxy stuck-on looking growth",
            "well-circumscribed with regular borders",
            "brown or tan colored",
            "slight itching when irritated",
        ],
        "contexts": [
            "age-related skin change",
            "trunk or back location",
        ],
        "severity_range": (1, 3),
    },
    "Dermatofibroma": {
        "symptoms": [
            "firm hard nodule under the skin",
            "dimple sign when pinched",
            "brownish or reddish color",
            "slightly tender to touch",
        ],
        "contexts": [
            "previous insect bite at the site",
            "lower extremity location",
        ],
        "severity_range": (1, 3),
    },
    "Vascular_Lesion": {
        "symptoms": [
            "red or purple patch on the skin",
            "lesion blanches with pressure",
            "clearly visible blood vessels",
            "no pain or itching",
        ],
        "contexts": [
            "present since birth or early childhood",
            "cosmetic concern",
        ],
        "severity_range": (1, 3),
    },
    # ==================== BRAIN (21-24) ====================
    "Glioma": {
        "symptoms": [
            "progressive headaches worse in the morning",
            "new onset seizures",
            "focal neurological deficits (e.g., weakness on one side)",
            "personality or cognitive changes",
            "nausea and vomiting from raised intracranial pressure",
            "visual disturbances",
            "speech difficulties",
        ],
        "contexts": [
            "previously healthy adult",
            "no prior history of seizures",
        ],
        "severity_range": (6, 10),
    },
    "Meningioma": {
        "symptoms": [
            "gradually worsening headaches",
            "seizures",
            "visual changes or visual field loss",
            "weakness in the extremities",
            "personality changes",
            "hearing loss on one side",
        ],
        "contexts": [
            "middle-aged female",
            "history of cranial radiation",
            "incidental finding on imaging",
        ],
        "severity_range": (3, 7),
    },
    "Pituitary_Tumor": {
        "symptoms": [
            "bitemporal hemianopsia (loss of peripheral vision)",
            "headaches",
            "amenorrhea or irregular menstrual periods",
            "galactorrhea",
            "fatigue and weight gain",
            "decreased libido",
            "visual acuity changes",
        ],
        "contexts": [
            "young adult female",
            "elevated prolactin levels on blood work",
            "Cushing syndrome features",
        ],
        "severity_range": (3, 7),
    },
    "No_Tumor": {
        "symptoms": [
            "tension-type headache",
            "migraine with aura",
            "non-specific dizziness",
            "anxiety-related symptoms",
            "no focal neurological deficits",
        ],
        "contexts": [
            "MRI ordered to rule out structural pathology",
            "normal neurological examination",
            "reassurance visit",
        ],
        "severity_range": (1, 3),
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Demographic templates
# ──────────────────────────────────────────────────────────────────────────────

AGES = list(range(18, 90))
GENDERS = ["male", "female"]
DURATIONS = [
    "for the past few days",
    "for about a week",
    "for two weeks",
    "for the past month",
    "over the last several months",
    "for approximately three months",
    "gradually over the past year",
]
SEVERITY_WORDS = {
    (1, 3): ["mild", "slight", "minimal"],
    (4, 6): ["moderate", "noticeable", "persistent"],
    (7, 8): ["severe", "significant", "debilitating"],
    (9, 10): ["critical", "life-threatening", "acute"],
}

TEMPLATES = [
    "A {age}-year-old {gender} presents with {symptoms}. {context}. "
    "Symptoms have been present {duration}. Severity assessed as {severity}.",

    "Patient is a {age}-year-old {gender} complaining of {symptoms} "
    "{duration}. Clinical context: {context}. Current severity: {severity}.",

    "{age} y/o {gender}. Chief complaint: {symptoms}. Duration: {duration}. "
    "Background: {context}. Severity: {severity}/10.",

    "History: {age}-year-old {gender} reports {symptoms} {duration}. "
    "Relevant context includes {context}. "
    "On a scale of 1-10 the patient rates severity as {severity}.",

    "Clinical presentation: {gender}, age {age}. {symptoms}. "
    "Onset: {duration}. Additional context: {context}. "
    "Assessed severity: {severity}/10.",
]


# ──────────────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────────────

def _severity_word(score: int) -> str:
    for (lo, hi), words in SEVERITY_WORDS.items():
        if lo <= score <= hi:
            return random.choice(words)
    return "moderate"


def generate_symptom_text(
    disease: str,
    rng: random.Random | None = None,
) -> str:
    """Generate a single synthetic symptom description for *disease*."""
    rng = rng or random
    info = DISEASE_SYMPTOM_MAP[disease]

    age = rng.choice(AGES)
    gender = rng.choice(GENDERS)
    duration = rng.choice(DURATIONS)
    severity_score = rng.randint(*info["severity_range"])
    severity = f"{_severity_word(severity_score)} ({severity_score}/10)"

    # Pick 2-5 symptoms
    n_symptoms = rng.randint(2, min(5, len(info["symptoms"])))
    symptoms = rng.sample(info["symptoms"], n_symptoms)
    symptom_str = ", ".join(symptoms[:-1]) + f", and {symptoms[-1]}"

    context = rng.choice(info["contexts"])
    template = rng.choice(TEMPLATES)

    return template.format(
        age=age,
        gender=gender,
        symptoms=symptom_str,
        context=context,
        duration=duration,
        severity=severity,
    )


def generate_multilabel_text(
    diseases: List[str],
    rng: random.Random | None = None,
) -> str:
    """
    Generate symptom text for a combination of co-occurring diseases.
    Merges symptoms from multiple diseases into one coherent description.
    """
    rng = rng or random
    all_symptoms = []
    all_contexts = []
    max_severity = 0

    for disease in diseases:
        info = DISEASE_SYMPTOM_MAP[disease]
        n = rng.randint(1, min(3, len(info["symptoms"])))
        all_symptoms.extend(rng.sample(info["symptoms"], n))
        all_contexts.append(rng.choice(info["contexts"]))
        max_severity = max(max_severity, rng.randint(*info["severity_range"]))

    # Deduplicate symptoms
    seen = set()
    unique_symptoms = []
    for s in all_symptoms:
        if s not in seen:
            unique_symptoms.append(s)
            seen.add(s)

    age = rng.choice(AGES)
    gender = rng.choice(GENDERS)
    duration = rng.choice(DURATIONS)
    severity = f"{_severity_word(max_severity)} ({max_severity}/10)"

    symptom_str = ", ".join(unique_symptoms[:-1]) + f", and {unique_symptoms[-1]}"
    context = "; ".join(all_contexts)
    template = rng.choice(TEMPLATES)

    return template.format(
        age=age,
        gender=gender,
        symptoms=symptom_str,
        context=context,
        duration=duration,
        severity=severity,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Multi-label pair generation
# ──────────────────────────────────────────────────────────────────────────────

# Common co-occurring disease pairs (within the same modality)
CO_OCCURRING_PAIRS = [
    # Chest
    ("Pneumonia", "Effusion"),
    ("Pneumonia", "Consolidation"),
    ("Cardiomegaly", "Edema"),
    ("Cardiomegaly", "Effusion"),
    ("Emphysema", "Pneumothorax"),
    ("Infiltration", "Atelectasis"),
    ("Mass", "Effusion"),
    ("Fibrosis", "Pleural_Thickening"),
]


def generate_dataset(
    samples_per_disease: int = 200,
    include_multilabel: bool = True,
    seed: int = 42,
) -> List[Tuple[str, List[str]]]:
    """
    Generate a full synthetic symptom dataset.

    Returns:
        List of (symptom_text, [disease_labels])
    """
    rng = random.Random(seed)
    dataset: List[Tuple[str, List[str]]] = []

    # Single-label samples
    for disease in DISEASE_SYMPTOM_MAP:
        for _ in range(samples_per_disease):
            text = generate_symptom_text(disease, rng)
            dataset.append((text, [disease]))

    # Multi-label samples (co-occurring)
    if include_multilabel:
        multi_per_pair = samples_per_disease // 4
        for d1, d2 in CO_OCCURRING_PAIRS:
            for _ in range(multi_per_pair):
                text = generate_multilabel_text([d1, d2], rng)
                dataset.append((text, [d1, d2]))

    rng.shuffle(dataset)
    return dataset


# ──────────────────────────────────────────────────────────────────────────────
# GPT augmentation prompt
# ──────────────────────────────────────────────────────────────────────────────

GPT_AUGMENTATION_PROMPT = """
You are a medical text generation assistant. Given the following disease label
and a seed symptom description, generate 5 diverse paraphrased versions of
the clinical presentation.

Requirements:
- Maintain medical accuracy
- Vary the writing style (clinical note, patient self-report, ER description)
- Randomise demographics (age, gender)
- Keep the core symptoms consistent with the disease
- Each variant should be 2-4 sentences

Disease: {disease}
Seed description: {seed_text}

Generate 5 variants:
""".strip()


def get_gpt_augmentation_prompt(disease: str, seed_text: str) -> str:
    """Return a prompt for GPT-based symptom augmentation."""
    return GPT_AUGMENTATION_PROMPT.format(disease=disease, seed_text=seed_text)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def save_to_csv(dataset: List[Tuple[str, List[str]]], output_path: str) -> None:
    """Save generated dataset to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "labels"])
        for text, labels in dataset:
            writer.writerow([text, "|".join(labels)])


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic symptoms")
    parser.add_argument("--output", type=str, default="data/symptoms.csv")
    parser.add_argument("--per-disease", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-multilabel", action="store_true")
    args = parser.parse_args()

    dataset = generate_dataset(
        samples_per_disease=args.per_disease,
        include_multilabel=not args.no_multilabel,
        seed=args.seed,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_to_csv(dataset, args.output)
    print(f"Generated {len(dataset)} samples → {args.output}")


if __name__ == "__main__":
    main()
