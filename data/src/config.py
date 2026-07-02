import os

# Define base paths dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'Data_Professional_Salary_Survey_Responses.xlsx')
DATA_PROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'processed')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

# Global parameters for modeling
RANDOM_STATE = 42