# -*- coding: utf-8 -*-
"""Unified ml1.py: Logistic Regression + GDP Scraper
python ml1.py --both
"""

import os
import sys
import warnings
import csv
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, accuracy_score, ConfusionMatrixDisplay)

warnings.filterwarnings('ignore')

# Optional imports for GDP scraping
try:
    import requests
    from bs4 import BeautifulSoup
    HAS_WEB_DEPS = True
except ImportError:
    HAS_WEB_DEPS = False


def safe_read_csv(path, synth_factory=None, info_name=None):
    """Try to read CSV at `path`. If missing, call `synth_factory()` to get
    a synthetic DataFrame. `info_name` is used in printed messages."""
    if os.path.exists(path):
        print(f"Loaded {info_name or path} from local file: {path}")
        return pd.read_csv(path)
    if synth_factory is not None:
        print(f"{path} not found — creating synthetic {info_name or path} for demo")
        return synth_factory()
    raise FileNotFoundError(path)


def synth_heart():
    np.random.seed(0)
    n = 300
    return pd.DataFrame({
        'Age': np.random.randint(29, 77, n),
        'Sex': np.random.choice([0, 1], n),
        'ChestPain': np.random.randint(1, 4, n),
        'RestBP': np.random.randint(90, 180, n),
        'Chol': np.random.randint(150, 300, n),
        'Fbs': np.random.choice([0, 1], n),
        'RestECG': np.random.randint(0, 2, n),
        'MaxHR': np.random.randint(90, 200, n),
        'ExAng': np.random.choice([0, 1], n),
        'Oldpeak': np.round(np.random.uniform(0.0, 6.0, n), 1),
        'Slope': np.random.randint(1, 3, n),
        'Ca': np.random.randint(0, 4, n),
        'Thal': np.random.choice([3, 6, 7], n),
        'AHD': np.random.choice(['Yes', 'No'], n)
    })


def synth_diabetes():
    np.random.seed(1)
    n = 500
    return pd.DataFrame({
        'Pregnancies': np.random.randint(0, 10, n),
        'Glucose': np.random.randint(70, 200, n),
        'BloodPressure': np.random.randint(50, 120, n),
        'SkinThickness': np.random.randint(10, 50, n),
        'Insulin': np.random.randint(15, 276, n),
        'BMI': np.round(np.random.uniform(18, 45, n), 1),
        'DiabetesPedigreeFunction': np.round(np.random.uniform(0.1, 2.5, n), 3),
        'Age': np.random.randint(21, 80, n),
        'Outcome': np.random.choice([0, 1], n, p=[0.65, 0.35])
    })


# ============================================================================
# GDP SCRAPER FUNCTIONS (requires requests and beautifulsoup4)
# ============================================================================

def fetch_page(url: str) -> str:
    """Fetch HTML from URL."""
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    resp.raise_for_status()
    return resp.text


def parse_top10(html: str) -> List[Tuple[str, str]]:
    """Parse GDP table and extract top 10 countries."""
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    target_table = None

    # Look for a table with a header containing 'Country' and 'GDP'
    for table in tables:
        th = table.find('th')
        if not th:
            continue
        header_text = th.get_text(separator=' ').strip()
        if 'Country' in header_text or 'Economy' in header_text:
            target_table = table
            break

    if target_table is None:
        # Fallback: try the first sortable wikitable
        for table in tables:
            if 'sortable' in (table.get('class') or []):
                target_table = table
                break

    if target_table is None:
        raise RuntimeError("Could not locate GDP table on the page")

    rows = target_table.find_all('tr')
    results: List[Tuple[str, str]] = []

    # Iterate rows and extract country and first numeric column
    for row in rows[1:]:
        cells = row.find_all(['th', 'td'])
        if len(cells) < 2:
            continue
        country = cells[0].get_text(strip=True)
        value = cells[1].get_text(strip=True)
        results.append((country, value))
        if len(results) >= 10:
            break

    return results


def save_gdp_csv(rows: List[Tuple[str, str]], path: str) -> None:
    """Save GDP top-10 to CSV."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'country', 'value'])
        for i, (country, value) in enumerate(rows, start=1):
            writer.writerow([i, country, value])


def run_gdp_scraper():
    """Fetch and save GDP (PPP) data."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    if not HAS_WEB_DEPS:
        print("Error: requests and beautifulsoup4 required for GDP scraper.")
        print("Install with: pip install requests beautifulsoup4")
        return

    url = "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(PPP)"
    out_csv = "gdp_top10.csv"

    print('Fetching GDP (PPP) page...')
    try:
        html = fetch_page(url)
    except Exception as e:
        print(f'Failed to fetch page: {e}')
        return

    try:
        top10 = parse_top10(html)
    except Exception as e:
        print(f'Failed to parse GDP table: {e}')
        return

    print('Top 10 countries by GDP (PPP):')
    for i, (country, value) in enumerate(top10, start=1):
        print(f"{i}. {country}: {value}")

    try:
        save_gdp_csv(top10, out_csv)
        print(f'✓ Saved top-10 CSV to {out_csv}')
    except Exception as e:
        print(f'Failed to save CSV: {e}')


# ============================================================================
# LOGISTIC REGRESSION FUNCTIONS
# ============================================================================

def run_logistic_regression():
    """Run logistic regression on diabetes dataset."""
    print('Starting logistic regression task...')

    # Attempt to read heart.csv (used earlier in the notebook)
    heart_df = safe_read_csv('heart.csv', synth_factory=synth_heart, info_name='heart dataset')
    print('\nHeart dataset preview:')
    print(heart_df.head())

    # Basic diagnostics
    print('\nHeart dataset info:')
    print(heart_df.info())

    # Demonstration: show a few summary stats
    if 'Age' in heart_df.columns:
        print('\nMean Age:', heart_df['Age'].mean())

    # Load Diabetes dataset (for logistic regression example)
    diabetes_df = safe_read_csv('Diabetes.csv', synth_factory=synth_diabetes, info_name='diabetes dataset')
    print('\nDiabetes dataset preview:')
    print(diabetes_df.head())

    # Prepare data for logistic regression
    if 'Outcome' not in diabetes_df.columns:
        print('No Outcome column in diabetes dataset; aborting logistic example.')
        return

    X = diabetes_df.drop('Outcome', axis=1)
    y = diabetes_df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    print('\nTraining Logistic Regression model...')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig = plt.figure(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Diabetes (Test)')
    out_file = 'ml1_confusion_matrix_diabetes.png'
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_file}")

    # Metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    fscore = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F-score: {fscore:.3f}")
    print(f"Accuracy: {accuracy:.3f}")


# ============================================================================
# MAIN DISPATCHER
# ============================================================================

def main():
    """Dispatcher: run tasks based on command-line args."""
    task = sys.argv[1] if len(sys.argv) > 1 else None

    if task == '--gdp':
        run_gdp_scraper()
    elif task == '--both':
        print("="*70)
        print("Running LOGISTIC REGRESSION")
        print("="*70)
        run_logistic_regression()
        print("\n" + "="*70)
        print("Running GDP SCRAPER")
        print("="*70)
        run_gdp_scraper()
    elif task is None or task == '--lr':
        run_logistic_regression()
    else:
        print(f"Unknown task: {task}")
        print("\nUsage:")
        print("  python ml1.py              # Run logistic regression (default)")
        print("  python ml1.py --gdp        # Run GDP scraper")
        print("  python ml1.py --both       # Run both tasks")
        sys.exit(1)


if __name__ == '__main__':
    main()

