from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATASET_PATH = Path('data/food_data.csv')
RESULTS_DIR = Path('data/mining_results')
MODELS_DIR = Path('data/mining_models')

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_COLUMNS = [
    'calories',
    'protein',
    'carb',
    'sugar',
    'fat',
    'saturated_fat',
    'fiber',
    'salt',
    'contains_milk',
    'contains_soy',
    'contains_gluten',
    'contains_wheat',
    'contains_peanut',
    'contains_hazelnut',
    'contains_almond',
    'contains_walnut',
    'contains_cashew',
    'contains_pistachio',
    'contains_egg',
    'contains_fish',
    'contains_sesame',
    'contains_mustard',
    'contains_celery',
    'contains_sulfites',
    'contains_lupin',
    'contains_crustaceans',
    'contains_molluscs',
]

TARGET_COLUMN = 'has_allergen_risk'


def build_models():
    models = {
        'naive_bayes': Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model', GaussianNB()),
            ],
        ),
        'knn': Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model', KNeighborsClassifier(n_neighbors=3)),
            ],
        ),
        'decision_tree': Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                (
                    'model',
                    DecisionTreeClassifier(
                        max_depth=5,
                        random_state=42,
                    ),
                ),
            ],
        ),
        'logistic_regression': Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                (
                    'model',
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ],
        ),
    }

    try:
        from xgboost import XGBClassifier

        models['xgboost'] = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                (
                    'model',
                    XGBClassifier(
                        n_estimators=100,
                        max_depth=3,
                        learning_rate=0.1,
                        eval_metric='logloss',
                        random_state=42,
                    ),
                ),
            ],
        )
    except ImportError:
        print('UYARI: xgboost kurulu değil. XGBoost modeli atlandı.')

    try:
        from lightgbm import LGBMClassifier

        models['lightgbm'] = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                (
                    'model',
                    LGBMClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        random_state=42,
                    ),
                ),
            ],
        )
    except ImportError:
        print('UYARI: lightgbm kurulu değil. LightGBM modeli atlandı.')

    return models


def normalize_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    boolean_columns = [
        column
        for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        if column.startswith('contains_') or column == TARGET_COLUMN
    ]

    for column in boolean_columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(
                {
                    'true': 1,
                    'false': 0,
                    '1': 1,
                    '0': 0,
                    'yes': 1,
                    'no': 0,
                    'nan': 0,
                    'none': 0,
                    '': 0,
                },
            )
        )

        df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(int)

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column
        for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f'CSV içinde eksik kolonlar var: {missing_columns}')

    df = normalize_boolean_columns(df)

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')

    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df


def validate_dataset(df: pd.DataFrame):
    if len(df) < 10:
        raise ValueError(
            'Model eğitimi için veri çok az. Önce en az 10 ürün ekleyelim. '
            'Asıl hedefimiz 50-100 ürün olacak.',
        )

    class_counts = df[TARGET_COLUMN].value_counts().to_dict()

    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError(
            'Target kolonunda sadece tek sınıf var. '
            'CSV içinde hem has_allergen_risk=0 hem de has_allergen_risk=1 '
            f'olan ürünler olmalı. Mevcut dağılım: {class_counts}',
        )

    min_class_count = df[TARGET_COLUMN].value_counts().min()

    if min_class_count < 2:
        raise ValueError(
            'Her sınıfta en az 2 ürün olmalı. '
            'Train/test ayrımı için risksiz ve riskli sınıflardan biraz daha eklemeliyiz. '
            f'Mevcut dağılım: {class_counts}',
        )


def calculate_roc_auc(model, x_test, y_test):
    try:
        if y_test.nunique() < 2:
            return None

        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(x_test)[:, 1]
            return roc_auc_score(y_test, y_proba)
    except Exception:
        return None

    return None


def save_confusion_matrix(model_name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['No Risk', 'Risk'],
    )

    display.plot()
    plt.title(f'Confusion Matrix - {model_name}')

    output_path = RESULTS_DIR / f'confusion_matrix_{model_name}.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return cm, output_path


def save_model_comparison_chart(results_df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.bar(results_df['model'], results_df['f1_score'])
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.title('Model Comparison by F1 Score')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    output_path = RESULTS_DIR / 'model_comparison_f1.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return output_path


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            'data/food_data.csv bulunamadı. '
            'Önce OCR endpoint ile birkaç ürünü dataset içine kaydetmelisin.',
        )

    df = pd.read_csv(DATASET_PATH)

    print('\nRAW DATA:')
    print(df)

    df = clean_dataset(df)

    print('\nCLEAN DATA:')
    print(df)

    print('\nSINIF DAĞILIMI:')
    print(df[TARGET_COLUMN].value_counts())

    validate_dataset(df)

    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = build_models()
    results = []

    for model_name, model in models.items():
        print(f'\nModel eğitiliyor: {model_name}')

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = calculate_roc_auc(model, x_test, y_test)

        cm, cm_path = save_confusion_matrix(model_name, y_test, y_pred)

        model_path = MODELS_DIR / f'{model_name}.joblib'
        joblib.dump(model, model_path)

        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]

        results.append(
            {
                'model': model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'tp': tp,
                'model_path': str(model_path),
                'confusion_matrix_path': str(cm_path),
            },
        )

    results_df = pd.DataFrame(results)
    metrics_path = RESULTS_DIR / 'metrics.csv'
    results_df.to_csv(metrics_path, index=False)

    chart_path = save_model_comparison_chart(results_df)

    print('\nMODEL SONUÇLARI:')
    print(results_df)

    print(f'\nMetrikler kaydedildi: {metrics_path}')
    print(f'Model karşılaştırma grafiği kaydedildi: {chart_path}')
    print(f'Modeller kaydedildi: {MODELS_DIR}')


if __name__ == '__main__':
    main()