import re
from collections import Counter
from typing import List, Dict
import string

def normalize_answer(s):
    def remove_(text):
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub("《", " ", text)
        text = re.sub("》", " ", text)
        text = re.sub("<", " ", text)
        text = re.sub(">", " ", text)
        text = re.sub("〈", " ", text)
        text = re.sub("〉", " ", text)
        text = re.sub(r"\(", " ", text)
        text = re.sub(r"\)", " ", text)
        text = re.sub("'", " ", text)
        text = re.sub("'", " ", text)
        return text

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction: str, ground_truths: List[str]) -> bool:
    return normalize_answer(prediction) in [normalize_answer(gt) for gt in ground_truths]

def compute_metrics(predicted_answer: str, ground_truth_answers: List[str]) -> Dict:
    em_score = exact_match_score(predicted_answer, ground_truth_answers)
    f1_scores = [f1_score(predicted_answer, gt) for gt in ground_truth_answers]
    max_f1 = max(f1_scores) if f1_scores else 0.0
    return {"em": 1.0 if em_score else 0.0, "f1": max_f1}