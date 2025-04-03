# Resume Rankeing Based on Job Description

## Overview

This project presents a structured approach to evaluating how well resumes align with a given job description (JD). The solution utilizes both keyword-based and contextual semantic methods to rank the top resumes relevant to a specific JD. The ranking system incorporates several recruiter-relevant features such as experience calculation, action verbs, location match, clarity, and online presence to provide a comprehensive scoring mechanism.

The project is divided into two key components:

1. **Data Selection (`selecting_the_data.py`)** – Selects a job description from a dataset and identifies the top 10 relevant resumes based on TF-IDF similarity.
2. **Resume Scoring and Ranking (`ranking_resumes.py`)** – Applies multiple feature-based scoring strategies to rank the selected resumes with respect to the chosen job description.

---

## Project Files

| File                      | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `selecting_the_data.py`   | Loads dataset from Hugging Face, selects 1 JD and 10 relevant resumes, saves as CSV |
| `ranking_resumes.py`      | Scores and ranks the 10 resumes based on multiple recruiter-like metrics    |
| `selected_jd_and_resumes.csv` | Auto-generated dataset containing the selected JD and associated resumes   |
| `ranked_scored_resumes.csv`   | Output file with resumes sorted by final computed score                  |

---

## Features Considered in Scoring

- **Contextual Similarity**: SBERT embeddings to measure semantic alignment between JD and resumes.
- **Skill Similarity**: TF-IDF-based vector similarity to match keyword overlap.
- **Experience Score**: Computed from date ranges extracted in resumes to calculate total professional experience.
- **Location Match**: Checks overlap between geographic entities in resume and JD.
- **Action Verb Strength**: Measures presence of strong leadership and impact verbs.
- **Achievement Score**: Detects presence of measurable accomplishments (e.g., %, $, teams, etc.).
- **Clarity Score**: Evaluated via average sentence length.
- **Link Score**: Checks presence and validity of external links (e.g., GitHub, portfolio).

---

## Installation

1. Clone the repository
```bash
git clone https://github.com/Priyarsha31/Resume-Ranker.git
cd Resume-Ranker
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate # for macOS
# On Windows use `venv\Scripts\activate`
```

### Make sure you have Python 3.8+ installed. Then install the required packages:

3. Install the required dependencies
```bash
pip install -r requirements.txt
```

4. You must also download the spaCy English language model with:
```bash
python -m spacy download en_core_web_sm
```

5. Run the scripts
   ```bash
   python selecting_the_data.py
   ```
   ```bash
   python ranking_resumes.py
   ```
