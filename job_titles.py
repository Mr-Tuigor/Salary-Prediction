
import pandas as pd

def get_job_titles():
    # Reload df2 to get original job titles before one-hot encoding
    df_original = pd.read_csv('Salary_Data.csv')
    # Clean and impute Job Title if necessary, as done in the notebook
    df_original['Job Title'] = df_original['Job Title'].fillna(df_original['Job Title'].mode()[0])
    unique_job_titles = sorted(df_original['Job Title'].unique().tolist())
    return unique_job_titles

JOB_TITLE_OPTIONS = get_job_titles()
