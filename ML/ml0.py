# -*- coding: utf-8 -*-
"""Cleaned ml0.py: Titanic Dataset Analysis & Logistic Regression

Converted from Colab notebook to a headless, production-ready script.
- Removes magics and inline pip installs
- Uses matplotlib Agg backend for headless plotting
- Replaces plt.show() with PNG saves
- Robust CSV loading with synthetic fallback
- Works in local and CI environments
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, recall_score, 
                             precision_score, confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


def safe_read_csv(path, synth_factory=None, info_name=None):
    """Try to read CSV at `path`. If missing, call `synth_factory()` to get
    a synthetic DataFrame. `info_name` is used in printed messages."""
    if os.path.exists(path):
        print(f"✓ Loaded {info_name or path} from local file: {path}")
        return pd.read_csv(path)
    if synth_factory is not None:
        print(f"✗ {path} not found — generating synthetic {info_name or path} for demo")
        return synth_factory()
    raise FileNotFoundError(path)


def synth_titanic():
    """Generate synthetic Titanic dataset for testing."""
    np.random.seed(42)
    n = 891
    return pd.DataFrame({
        'PassengerId': range(1, n + 1),
        'Pclass': np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
        'Age': np.random.normal(29, 12, n).clip(0, 80),
        'SibSp': np.random.randint(0, 6, n),
        'Parch': np.random.randint(0, 6, n),
        'Fare': np.random.exponential(32, n).clip(0, 512),
        'Survived': np.random.binomial(1, 0.38, n)
    })


def main():
    print("="*70)
    print("ML0: Titanic Dataset Analysis & Logistic Regression")
    print("="*70)

    # Load Titanic dataset
    df = safe_read_csv('titanic_dataset.csv', synth_factory=synth_titanic, 
                       info_name='Titanic dataset')
    
    print("\nDataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())

    # 1. Analyze missing values
    print("\n" + "="*70)
    print("MISSING VALUES ANALYSIS")
    print("="*70)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=True, cmap='viridis', ax=ax)
    plt.title('Missing Values Heatmap - Titanic Dataset')
    plt.tight_layout()
    plt.savefig('ml0_missing_values_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: ml0_missing_values_heatmap.png")

    # a) Cabin missing count
    if 'Cabin' in df.columns:
        cabin_null = df['Cabin'].isnull().sum()
        print(f"\nCabin null count: {cabin_null}")

    # b) Cabin vs Survived
    if 'Cabin' in df.columns and 'Survived' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Create binary Cabin indicator (has cabin or not)
        df['HasCabin'] = (~df['Cabin'].isnull()).astype(int)
        sns.countplot(x='HasCabin', hue='Survived', data=df, ax=ax)
        plt.title('Cabin Presence vs Survival')
        plt.xlabel('Has Cabin (0=No, 1=Yes)')
        plt.tight_layout()
        plt.savefig('ml0_cabin_vs_survived.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: ml0_cabin_vs_survived.png")

    # c) Cabin vs Sex (if available)
    if 'Cabin' in df.columns and 'Sex' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='HasCabin', hue='Sex', data=df, ax=ax)
        plt.title('Cabin Presence vs Sex')
        plt.xlabel('Has Cabin (0=No, 1=Yes)')
        plt.tight_layout()
        plt.savefig('ml0_cabin_vs_sex.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: ml0_cabin_vs_sex.png")

    # d) Cabin vs Pclass
    if 'Cabin' in df.columns and 'Pclass' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='HasCabin', hue='Pclass', data=df, ax=ax)
        plt.title('Cabin Presence vs Passenger Class')
        plt.xlabel('Has Cabin (0=No, 1=Yes)')
        plt.tight_layout()
        plt.savefig('ml0_cabin_vs_pclass.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: ml0_cabin_vs_pclass.png")

    # 2. Age distribution
    if 'Age' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, bins=40, x='Age', kde=True, ax=ax)
        plt.title('Age Distribution - Titanic')
        plt.tight_layout()
        plt.savefig('ml0_age_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: ml0_age_distribution.png")

    # 3. Siblings/Spouses distribution
    if 'SibSp' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='SibSp', data=df, ax=ax)
        plt.title('Siblings/Spouses Count Distribution')
        plt.tight_layout()
        plt.savefig('ml0_sibsp_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: ml0_sibsp_distribution.png")

    # 4. Fare distribution
    if 'Fare' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Fare'], bins=40, kde=False, ax=ax)
        plt.title('Fare Distribution - Titanic')
        plt.tight_layout()
        plt.savefig('ml0_fare_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: ml0_fare_distribution.png")

    # 5. Age vs Passenger Class
    if 'Age' in df.columns and 'Pclass' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Pclass', y='Age', data=df, ax=ax)
        plt.title('Age Distribution by Passenger Class')
        plt.tight_layout()
        plt.savefig('ml0_age_by_pclass.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: ml0_age_by_pclass.png")

    # 6. Data Cleaning
    print("\n" + "="*70)
    print("DATA CLEANING")
    print("="*70)

    # Handle Age
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        age_null_before = df['Age'].isnull().sum()
        df['Age'] = df['Age'].fillna(df['Age'].mean())
        age_null_after = df['Age'].isnull().sum()
        print(f"Age: filled {age_null_before} nulls → {age_null_after} remaining")

    # Drop irrelevant columns
    cols_to_drop = ['Cabin', 'Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId']
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    print("\nDataset info after cleaning:")
    print(df.info())

    # 7. Logistic Regression
    print("\n" + "="*70)
    print("LOGISTIC REGRESSION MODEL")
    print("="*70)

    if 'Survived' not in df.columns:
        print("✗ No 'Survived' column found; skipping logistic regression.")
        return

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Train model
    logreg = LogisticRegression(max_iter=1000)
    print("\nTraining Logistic Regression model...")
    logreg.fit(X_train, y_train)

    # Predictions
    train_pred = logreg.predict(X_train)
    test_pred = logreg.predict(X_test)

    train_acc = accuracy_score(train_pred, y_train)
    test_acc = accuracy_score(test_pred, y_test)

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")

    # Metrics
    precision = precision_score(y_test, test_pred)
    recall = recall_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=['Not Survived', 'Survived']
    )
    fig = plt.figure(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Titanic Survival Prediction (Test)')
    plt.tight_layout()
    plt.savefig('ml0_confusion_matrix_titanic.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: ml0_confusion_matrix_titanic.png")

    print("\n" + "="*70)
    print("ML0 COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()

