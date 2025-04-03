# This script is to randomly select a job description from the Hugging Face dataset
# and then select top 10 relevant resumes from the dataset to that JD and save it
# as a new csv file "selected_jd_And_resumes.csv"


# Import all the necessary packages
import pandas as pd
import re
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to clean the text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

# Loading the dataset from Hugging Face
print("Loading dataset from Hugging Face...")
raw_data = load_dataset("cnamuangtoun/resume-job-description-fit", split='train')
data_df = pd.DataFrame(raw_data)

# Cleaning and dropping the mising data 
data_df = data_df.dropna(subset=['resume_text', 'job_description_text'])
data_df['cleaned_resume'] = data_df['resume_text'].apply(clean_text)
data_df['cleaned_jd'] = data_df['job_description_text'].apply(clean_text)

# Removing duplicate resumes
data_df = data_df.drop_duplicates(subset='cleaned_resume')

# Selecting a Job Description randomly
print("Selecting one job description randomly")
chosen_jd_row = data_df.sample(n=1, random_state=21).reset_index(drop=True)
chosen_jd_text = chosen_jd_row.loc[0, 'cleaned_jd']

# Ensuring that the JD isn't matched as a resume
data_df = data_df[data_df['cleaned_jd'] != chosen_jd_text]

# Vectorizing the resumes using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
all_resume_texts = data_df["clean_resume"].tolist()
tfidf_matrix = vectorizer.fit_transform(all_resume_texts + [chosen_jd_text])

# Calculating the cosine similarity of the resumes with the chosen JD.
resume_vectors = tfidf_matrix[:-1]
jd_vector = tfidf_matrix[-1]
similarity_scores = cosine_similarity(resume_vectors, jd_vector).flatten()

# Sorting the resumes based on the similarity and picking up top 10 resumes
sorted_indices = similarity_scores.argsort()[::-1]
selected_resume_indices = []
seen_resumes = set()

for idx in sorted_indices:
    this_resume = data_df.iloc[idx]["clean_resume"]
    if this_resume not in seen_resumes:
        seen_resumes.add(this_resume)
        selected_resume_indices.append(idx)
    if len(selected_resume_indices) == 10:
        break

top_10_resumes_df = data_df.iloc[selected_resume_indices][['cleaned_resume']].reset_index(drop=True)

# To add the Job Description at the top of the file
jd_clean_df = pd.DataFrame({'cleaned_resume': [chosen_jd_text]})
final_data_df = pd.concat([jd_clean_df, top_10_resumes_df], ignore_index=True)

# Saving the resumes and JD as a CSV file
final_data_df.to_csv("selected_jd_and_resumes.csv", index=False)
print("Saved 1 cleaned JD followed by 10 cleaned resumes to 'selected_jd_and_resumes.csv'")
