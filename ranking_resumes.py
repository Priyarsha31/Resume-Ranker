# This is the main script that evaluates a final score to each of the 10 resumes
# against the JD and rank them, finally store the results in a new CSV file
# "ranked_scored_resumes.csv"

# Import the necessary packages
import pandas as pd
import numpy as np
import re
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from dateparser import parse

# Try to load the spaCy model and raise an exception accordingly
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")

# Loading the SBERT(Sentence Transformer) model for capturing semantic similarity 
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Cleaning the text
def clean_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Extracting the date ranges from the resumes
def extract_date_periods(text):
    date_patterns = re.findall(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)?\s?20\d{2}).{1,5}((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)?\s?20\d{2}|present)', text, flags=re.IGNORECASE)
    date_ranges = []
    for start, end in date_patterns:
        start_date = parse(start)
        end_date = datetime.now() if re.search(r'present', end, re.IGNORECASE) else parse(end)
        if start_date and end_date and start_date < end_date:
            date_ranges.append((start_date, end_date))
    return date_ranges

# Combining the overlapping time ranges
def merge_periods(periods):
    periods = sorted(periods, key=lambda x: x[0])
    merged = []
    for start, end in periods:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged

# Calculating the total experience score from the obtained date ranges
def calculate_experience_score(text):
    periods = merge_periods(extract_date_periods(text))
    total_months = sum((end.year - start.year) * 12 + (end.month - start.month) for start, end in periods)
    total_years = total_months // 12
    return min(total_years / 10, 1.0)  # normalizing to 0–1 range

# To see if the location in JD and resume overlap if any
def get_location(text):
    doc = nlp(text)
    return set(ent.text.lower() for ent in doc.ents if ent.label_ == "GPE")

# Calculating the location score
def score_location(jd_text, resume_text):
    return 1.0 if get_location(jd_text) & get_location(resume_text) else 0.0

# Calculating the action verb score based on a few pre-defined verbs
def score_action_verbs(text):
    strong_verbs = ['led', 'built', 'created', 'developed', 'implemented', 'designed', 'achieved']
    return sum(verb in text.lower() for verb in strong_verbs) / len(strong_verbs)

# Calculating the achievements score
def score_achievements(text):
    return min(len(re.findall(r'\b(\d+%|\$\d+|\d+ projects|\d+ teams)\b', text)) / 5.0, 1.0)

# Calculating the clarity score
def score_clarity(text):
    sentences = re.split(r'[.!?]', text)
    avg_words = np.mean([len(s.split()) for s in sentences if len(s.strip()) > 0])
    return 1.0 if avg_words < 25 else 0.5 if avg_words < 40 else 0.3

# Checking if the links are working correctly(if any) and calculating the link score
def score_links(text):
    urls = re.findall(r'(https?://\S+)', text)
    working = 0
    for url in urls:
        try:
            if requests.head(url, timeout=8).status_code == 200:
                working += 1
        except:
            continue
    return min(working / 2.0, 1.0)

# Calculating the keyword-based and semantic similarity score using SBERT
def contextual_similarity(resumes, jd_text):
    jd_vec = sbert_model.encode(jd_text, convert_to_tensor=True)
    truncated_resumes = resumes['clean_resume'].apply(lambda x: x[:600])
    resume_vecs = sbert_model.encode(truncated_resumes.tolist(), convert_to_tensor=True)
    return util.cos_sim(resume_vecs, jd_vec).cpu().numpy().flatten()

# Calculating the skill similarity score using TF-IDF
def skill_similarity(resumes, jd_text):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    texts = resumes['clean_resume'].tolist() + [jd_text]
    tfidf_matrix = vectorizer.fit_transform(texts)
    return cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1]).flatten()

# Ranking the resumes after calculating the final score
def rank_resumes(jd_text, resumes):
    resumes = resumes.copy()
    resumes['clean_resume'] = resumes['resume_text'].apply(clean_text)
    jd_clean = clean_text(jd_text)

    resumes['context_score'] = contextual_similarity(resumes, jd_clean)
    resumes['skill_score'] = skill_similarity(resumes, jd_clean)
    resumes['experience_score'] = resumes['resume_text'].apply(calculate_experience_score)
    resumes['location_score'] = resumes['resume_text'].apply(lambda x: score_location(jd_text, x))
    resumes['verb_score'] = resumes['resume_text'].apply(score_action_verbs)
    resumes['impact_score'] = resumes['resume_text'].apply(score_achievements)
    resumes['clarity_score'] = resumes['resume_text'].apply(score_clarity)
    resumes['link_score'] = resumes['resume_text'].apply(score_links)

    resumes['final_score'] = (
        0.30 * resumes['context_score'] +
        0.20 * resumes['skill_score'] +
        0.15 * resumes['experience_score'] +
        0.10 * resumes['location_score'] +
        0.10 * resumes['verb_score'] +
        0.05 * resumes['impact_score'] +
        0.05 * resumes['clarity_score'] +
        0.05 * resumes['link_score']
    )

    return resumes.sort_values(by='final_score', ascending=False)



if __name__ == "__main__":
    data_df = pd.read_csv("selected_jd_and_resumes.csv")  # Must contain 'resume_text' and 'job_description_text'
    jd_description_text = data_df['cleaned_resume'].iloc[0]
    resumes_df = data_df[['cleaned_resume']].iloc[1:].reset_index(drop=True)
    resumes_df = resumes_df.rename(columns={'cleaned_resume': 'resume_text'})
    ranked_resumes = rank_resumes(jd_description_text, resumes_df)
    ranked_resumes[['resume_text', 'final_score']].to_csv("ranked_scored_resumes.csv", index=False)
    print("Done! Top resumes are saved to 'ranked_scored_resumes.csv'")
