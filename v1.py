"""
NASA Turbofan Engine Predictive Maintenance - Complete ML Pipeline
===================================================================

This notebook implements end-to-end predictive maintenance for aircraft engines
using the NASA C-MAPSS (Turbofan Engine Degradation) dataset.

Dataset: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
or: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

Project Goal: Predict Remaining Useful Life (RUL) and identify engines 
that will fail within 30 cycles.
"""

# ============================================================================
# SECTION 1: SETUP AND DATA LOADING
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             classification_report, confusion_matrix, roc_auc_score,
                             precision_score, recall_score, f1_score, roc_curve)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("NASA TURBOFAN PREDICTIVE MAINTENANCE PROJECT")
print("=" * 70)

# ============================================================================
# FUNCTION: Load and Prepare Data
# ============================================================================

def load_cmapss_data(dataset_number=1):
    """
    Load NASA C-MAPSS dataset
    
    Parameters:
    -----------
    dataset_number : int (1-4)
        Which dataset to load:
        FD001: Simple (one operating condition, one fault mode)
        FD002: Complex operating conditions
        FD003: Simple with different fault mode
        FD004: Most complex (multiple conditions and fault modes)
    
    Returns:
    --------
    train_df, test_df, rul_df : DataFrames
    """
    
    # Column names
    index_names = ['unit', 'cycle']
    setting_names = ['setting1', 'setting2', 'setting3']
    sensor_names = [f'sensor{i}' for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    
    # File paths (adjust based on your download location)
    train_file = f'train_FD00{dataset_number}.txt'
    test_file = f'test_FD00{dataset_number}.txt'
    rul_file = f'RUL_FD00{dataset_number}.txt'
    
    try:
        # Load data
        train = pd.read_csv(train_file, sep=r'\s+', header=None, names=col_names)
        test = pd.read_csv(test_file, sep=r'\s+', header=None, names=col_names)
        rul = pd.read_csv(rul_file, sep=r'\s+', header=None, names=['RUL'])
        
        print(f"\n✓ Successfully loaded FD00{dataset_number} dataset")
        print(f"  Training engines: {train['unit'].nunique()}")
        print(f"  Test engines: {test['unit'].nunique()}")
        print(f"  Training cycles: {len(train):,}")
        
        return train, test, rul
        
    except FileNotFoundError:
        print(f"\n✗ ERROR: Dataset files not found!")
        print(f"\nPlease download the NASA C-MAPSS dataset from:")
        print("  Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
        print("  or NASA: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
        print(f"\nExpected files in current directory:")
        print(f"  - {train_file}")
        print(f"  - {test_file}")
        print(f"  - {rul_file}")
        return None, None, None

# ============================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS
# ============================================================================

def explore_data(train_df):
    """Perform exploratory data analysis"""
    
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    
    # Basic statistics
    print("\n1. DATASET SHAPE")
    print(f"   Rows: {len(train_df):,}")
    print(f"   Columns: {len(train_df.columns)}")
    
    print("\n2. SAMPLE DATA (First 5 rows)")
    print(train_df.head())
    
    print("\n3. DATA TYPES")
    print(train_df.dtypes.value_counts())
    
    print("\n4. MISSING VALUES")
    missing = train_df.isnull().sum()
    if missing.sum() == 0:
        print("   ✓ No missing values detected")
    else:
        print(missing[missing > 0])
    
    print("\n5. ENGINE LIFECYCLE STATISTICS")
    lifecycle = train_df.groupby('unit')['cycle'].max()
    print(f"   Mean lifecycle: {lifecycle.mean():.1f} cycles")
    print(f"   Min lifecycle: {lifecycle.min()} cycles")
    print(f"   Max lifecycle: {lifecycle.max()} cycles")
    print(f"   Std deviation: {lifecycle.std():.1f} cycles")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Lifecycle distribution
    axes[0, 0].hist(lifecycle, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Cycles to Failure')
    axes[0, 0].set_ylabel('Number of Engines')
    axes[0, 0].set_title('Distribution of Engine Lifecycles')
    axes[0, 0].axvline(lifecycle.mean(), color='red', linestyle='--', 
                       label=f'Mean: {lifecycle.mean():.0f}')
    axes[0, 0].legend()
    
    # 2. Sample engine degradation (first 5 engines)
    for unit in train_df['unit'].unique()[:5]:
        engine_data = train_df[train_df['unit'] == unit]
        axes[0, 1].plot(engine_data['cycle'], engine_data['sensor2'], 
                       alpha=0.6, label=f'Engine {unit}')
    axes[0, 1].set_xlabel('Cycle')
    axes[0, 1].set_ylabel('Sensor 2 Reading')
    axes[0, 1].set_title('Sample Sensor Degradation Patterns')
    axes[0, 1].legend()
    
    # 3. Sensor correlation heatmap (sample)
    sensor_cols = [col for col in train_df.columns if 'sensor' in col][:10]
    corr_matrix = train_df[sensor_cols].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=axes[1, 0],
                cbar_kws={'label': 'Correlation'})
    axes[1, 0].set_title('Sensor Correlation Matrix (First 10 Sensors)')
    
    # 4. Operating settings distribution
    axes[1, 1].scatter(train_df['setting1'], train_df['setting2'], 
                      alpha=0.1, s=1)
    axes[1, 1].set_xlabel('Setting 1')
    axes[1, 1].set_ylabel('Setting 2')
    axes[1, 1].set_title('Operating Conditions Distribution')
    
    plt.tight_layout()
    plt.savefig('eda_visualization.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualizations saved as 'eda_visualization.png'")
    plt.show()

# ============================================================================
# SECTION 3: FEATURE ENGINEERING
# ============================================================================

def add_rul_target(df):
    """
    Add Remaining Useful Life (RUL) as target variable
    For each engine, RUL = max_cycle - current_cycle
    """
    # Calculate max cycle for each engine
    max_cycles = df.groupby('unit')['cycle'].max().reset_index()
    max_cycles.columns = ['unit', 'max_cycle']
    
    # Merge and calculate RUL
    df = df.merge(max_cycles, on='unit', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df = df.drop('max_cycle', axis=1)
    
    return df

def add_rolling_features(df, window_sizes=[5, 10, 20]):
    """
    Add rolling statistics for sensor readings
    Captures degradation trends over time
    """
    sensor_cols = [col for col in df.columns if 'sensor' in col]
    
    for window in window_sizes:
        for sensor in sensor_cols:
            # Rolling mean
            df[f'{sensor}_rolling_mean_{window}'] = df.groupby('unit')[sensor]\
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            
            # Rolling std (volatility)
            df[f'{sensor}_rolling_std_{window}'] = df.groupby('unit')[sensor]\
                .transform(lambda x: x.rolling(window, min_periods=1).std()).fillna(0)
    
    return df

def add_degradation_features(df):
    """
    Add features capturing degradation rate
    """
    sensor_cols = [col for col in df.columns if 'sensor' in col and 'rolling' not in col]
    
    # Calculate max cycle once for all sensors
    max_cycle_per_unit = df.groupby('unit')['cycle'].max().reset_index()
    max_cycle_per_unit.columns = ['unit', 'max_cycle']
    df = df.merge(max_cycle_per_unit, on='unit', how='left')
    
    for sensor in sensor_cols:
        # Difference from first cycle (degradation from baseline)
        df[f'{sensor}_diff_from_start'] = df.groupby('unit')[sensor]\
            .transform(lambda x: x - x.iloc[0])
        
        # Cycle-normalized sensor reading (accounts for different lifecycles)
        df[f'{sensor}_normalized'] = df[sensor] / (df['max_cycle'] + 1)
    
    # Drop max_cycle column after using it
    df = df.drop('max_cycle', axis=1)
    
    return df

def engineer_features(df, window_sizes=[5, 10]):
    """
    Complete feature engineering pipeline
    """
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)
    
    print("\n1. Adding RUL target variable...")
    df = add_rul_target(df)
    
    print("2. Adding rolling statistics (windows: {})...".format(window_sizes))
    df = add_rolling_features(df, window_sizes)
    
    print("3. Adding degradation features...")
    df = add_degradation_features(df)
    
    # Add binary classification target (failure within 30 cycles)
    df['failure_within_30'] = (df['RUL'] <= 30).astype(int)
    
    print(f"\n✓ Feature engineering complete!")
    print(f"  Original features: 26")
    print(f"  Total features now: {len(df.columns)}")
    
    return df

# ============================================================================
# SECTION 4: DATA PREPARATION
# ============================================================================

def prepare_data_for_modeling(train_df, test_df, test_rul):
    """
    Prepare train and test sets for modeling
    """
    print("\n" + "=" * 70)
    print("DATA PREPARATION FOR MODELING")
    print("=" * 70)
    
    # Feature engineer both sets
    train_processed = engineer_features(train_df.copy())
    test_processed = engineer_features(test_df.copy())
    
    # For test set, we need to get the last cycle for each engine
    # and merge with the true RUL values
    test_last_cycles = test_processed.groupby('unit')['cycle'].max().reset_index()
    test_last_cycles.columns = ['unit', 'last_cycle']
    test_last_cycles = test_last_cycles.sort_values('unit').reset_index(drop=True)
    test_last_cycles['RUL_true'] = test_rul['RUL'].values
    
    # Get last cycle data for each engine in test set
    test_final = test_processed.merge(
        test_last_cycles[['unit', 'last_cycle']], 
        on='unit', 
        how='left'
    )
    test_final = test_final[test_final['cycle'] == test_final['last_cycle']].copy()
    test_final = test_final.drop('last_cycle', axis=1)
    test_final = test_final.sort_values('unit').reset_index(drop=True)
    test_final['RUL_true'] = test_rul['RUL'].values
    
    # Select features (exclude targets and identifiers)
    exclude_cols = ['unit', 'cycle', 'RUL', 'failure_within_30']
    feature_cols = [col for col in train_processed.columns if col not in exclude_cols]
    
    # Prepare training data
    X_train = train_processed[feature_cols].copy()
    y_train_rul = train_processed['RUL'].copy()
    y_train_failure = train_processed['failure_within_30'].copy()
    
    # Prepare test data
    X_test = test_final[feature_cols].copy()
    y_test_rul = test_final['RUL_true'].copy()
    
    # For test failure prediction, create target based on true RUL
    y_test_failure = (test_final['RUL_true'] <= 30).astype(int)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n✓ Data preparation complete!")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Features: {len(feature_cols)}")
    
    return (X_train_scaled, X_test_scaled, y_train_rul, y_test_rul, 
            y_train_failure, y_test_failure, feature_cols, scaler)

# ============================================================================
# SECTION 5: RUL PREDICTION (REGRESSION)
# ============================================================================

def train_rul_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate RUL prediction models
    """
    print("\n" + "=" * 70)
    print("RUL PREDICTION MODEL TRAINING")
    print("=" * 70)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, 
                                                random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, 
                                                        max_depth=5, 
                                                        random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_test = r2_score(y_test, y_pred_test)
        
        results[name] = {
            'model': model,
            'mae_train': mae_train,
            'mae_test': mae_test,
            'rmse_test': rmse_test,
            'r2_test': r2_test,
            'y_pred_test': y_pred_test
        }
        
        print(f"  MAE (test): {mae_test:.2f} cycles")
        print(f"  RMSE (test): {rmse_test:.2f} cycles")
        print(f"  R² (test): {r2_test:.3f}")
    
    # Select best model
    best_model_name = min(results.keys(), key=lambda x: results[x]['mae_test'])
    best_model = results[best_model_name]['model']
    
    print(f"\n✓ Best model: {best_model_name} (MAE: {results[best_model_name]['mae_test']:.2f})")
    
    return results, best_model_name

def visualize_rul_results(results, y_test, best_model_name):
    """
    Visualize RUL prediction results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Prediction vs Actual
    best_pred = results[best_model_name]['y_pred_test']
    axes[0, 0].scatter(y_test, best_pred, alpha=0.5, s=20)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                    'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('True RUL (cycles)')
    axes[0, 0].set_ylabel('Predicted RUL (cycles)')
    axes[0, 0].set_title(f'RUL Prediction: {best_model_name}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Error distribution
    errors = y_test - best_pred
    axes[0, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='red', linestyle='--', label='Zero Error')
    axes[0, 1].set_xlabel('Prediction Error (cycles)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Error Distribution')
    axes[0, 1].legend()
    
    # 3. Model comparison
    model_names = list(results.keys())
    mae_scores = [results[name]['mae_test'] for name in model_names]
    rmse_scores = [results[name]['rmse_test'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    axes[1, 0].bar(x - width/2, mae_scores, width, label='MAE', alpha=0.8)
    axes[1, 0].bar(x + width/2, rmse_scores, width, label='RMSE', alpha=0.8)
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Error (cycles)')
    axes[1, 0].set_title('Model Performance Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Residuals plot
    axes[1, 1].scatter(best_pred, errors, alpha=0.5, s=20)
    axes[1, 1].axhline(0, color='red', linestyle='--', label='Zero Error')
    axes[1, 1].set_xlabel('Predicted RUL (cycles)')
    axes[1, 1].set_ylabel('Residuals (True - Predicted)')
    axes[1, 1].set_title('Residuals Plot')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rul_model_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ RUL visualization saved as 'rul_model_results.png'")
    plt.show()

# ============================================================================
# SECTION 6: FAILURE PREDICTION (CLASSIFICATION)
# ============================================================================

def train_failure_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate failure prediction models
    """
    print("\n" + "=" * 70)
    print("FAILURE PREDICTION MODEL TRAINING")
    print("=" * 70)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15,
                                                 random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100,
                                                         max_depth=5,
                                                         random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        roc_auc = roc_auc_score(y_test, y_pred_proba_test)
        
        results[name] = {
            'model': model,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred_test': y_pred_test,
            'y_pred_proba_test': y_pred_proba_test
        }
        
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        print(f"  ROC-AUC: {roc_auc:.3f}")
    
    # Select best model (by F1 score)
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]['model']
    
    print(f"\n✓ Best model: {best_model_name} (F1: {results[best_model_name]['f1']:.3f})")
    
    return results, best_model_name

def visualize_failure_results(results, y_test, best_model_name):
    """
    Visualize failure prediction results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    best_pred = results[best_model_name]['y_pred_test']
    best_proba = results[best_model_name]['y_pred_proba_test']
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, best_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['No Failure', 'Failure'],
                yticklabels=['No Failure', 'Failure'])
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_title(f'Confusion Matrix: {best_model_name}')
    
    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, best_proba)
    axes[0, 1].plot(fpr, tpr, label=f'{best_model_name} (AUC = {results[best_model_name]["roc_auc"]:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Model comparison
    model_names = list(results.keys())
    precision_scores = [results[name]['precision'] for name in model_names]
    recall_scores = [results[name]['recall'] for name in model_names]
    f1_scores = [results[name]['f1'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.25
    axes[1, 0].bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
    axes[1, 0].bar(x, recall_scores, width, label='Recall', alpha=0.8)
    axes[1, 0].bar(x + width, f1_scores, width, label='F1 Score', alpha=0.8)
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Model Performance Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim([0, 1])
    
    # 4. Probability distribution
    axes[1, 1].hist(best_proba[y_test == 0], bins=30, alpha=0.5, 
                    label='No Failure', color='green', edgecolor='black')
    axes[1, 1].hist(best_proba[y_test == 1], bins=30, alpha=0.5, 
                    label='Failure', color='red', edgecolor='black')
    axes[1, 1].axvline(0.5, color='blue', linestyle='--', label='Threshold (0.5)')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Probability Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('failure_model_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Failure prediction visualization saved as 'failure_model_results.png'")
    plt.show()

# ============================================================================
# SECTION 7: FEATURE IMPORTANCE
# ============================================================================

def visualize_feature_importance(model, feature_names, task_name='RUL', top_n=20):
    """
    Visualize feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("Model does not have feature_importances_ attribute")
        return
    
    # Get top N features
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_importances, align='center')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Features for {task_name} Prediction')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'feature_importance_{task_name.lower()}.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Feature importance saved as 'feature_importance_{task_name.lower()}.png'")
    plt.show()

# ============================================================================
# SECTION 8: PRODUCTION MONITORING SIMULATION
# ============================================================================

def simulate_production_monitoring(rul_model, failure_model, X_test, y_test_rul, 
                                    y_test_failure, feature_names):
    """
    Simulate production monitoring and alerting
    """
    print("\n" + "=" * 70)
    print("PRODUCTION MONITORING SIMULATION")
    print("=" * 70)
    
    # Predictions
    rul_predictions = rul_model.predict(X_test)
    failure_predictions = failure_model.predict(X_test)
    failure_probabilities = failure_model.predict_proba(X_test)[:, 1]
    
    # Create monitoring dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. RUL predictions over time (simulated)
    axes[0, 0].plot(range(len(rul_predictions[:50])), rul_predictions[:50], 
                    'b-', alpha=0.7, label='Predicted RUL')
    axes[0, 0].plot(range(len(y_test_rul[:50])), y_test_rul[:50], 
                    'r--', alpha=0.7, label='True RUL')
    axes[0, 0].axhline(30, color='orange', linestyle=':', label='Alert Threshold (30 cycles)')
    axes[0, 0].set_xlabel('Engine Index')
    axes[0, 0].set_ylabel('RUL (cycles)')
    axes[0, 0].set_title('RUL Predictions (Sample)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Failure alerts
    alert_threshold = 0.7
    high_risk = failure_probabilities >= alert_threshold
    axes[0, 1].scatter(range(len(failure_probabilities)), failure_probabilities, 
                       c=high_risk, cmap='RdYlGn', alpha=0.6, s=30)
    axes[0, 1].axhline(alert_threshold, color='red', linestyle='--', 
                       label=f'Alert Threshold ({alert_threshold})')
    axes[0, 1].set_xlabel('Engine Index')
    axes[0, 1].set_ylabel('Failure Probability')
    axes[0, 1].set_title('Failure Risk Alerts')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error tracking
    rul_errors = np.abs(y_test_rul - rul_predictions)
    axes[1, 0].plot(range(len(rul_errors)), rul_errors, 'g-', alpha=0.6)
    axes[1, 0].axhline(rul_errors.mean(), color='red', linestyle='--', 
                       label=f'Mean Error: {rul_errors.mean():.2f}')
    axes[1, 0].set_xlabel('Engine Index')
    axes[1, 0].set_ylabel('Absolute Error (cycles)')
    axes[1, 0].set_title('Prediction Error Tracking')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Alert summary
    alert_counts = {
        'High Risk (RUL < 30)': np.sum(rul_predictions < 30),
        'Very High Risk (RUL < 15)': np.sum(rul_predictions < 15),
        'Failure Prob > 0.7': np.sum(failure_probabilities > 0.7),
        'Failure Prob > 0.9': np.sum(failure_probabilities > 0.9)
    }
    
    axes[1, 1].barh(list(alert_counts.keys()), list(alert_counts.values()), 
                    color=['orange', 'red', 'darkred', 'maroon'])
    axes[1, 1].set_xlabel('Number of Engines')
    axes[1, 1].set_title('Alert Summary')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('production_monitoring_dashboard.png', dpi=300, bbox_inches='tight')
    print("\n✓ Production monitoring dashboard saved as 'production_monitoring_dashboard.png'")
    plt.show()
    
    # Print summary
    print(f"\n📊 Monitoring Summary:")
    print(f"  Engines with RUL < 30 cycles: {np.sum(rul_predictions < 30)}")
    print(f"  Engines with RUL < 15 cycles: {np.sum(rul_predictions < 15)}")
    print(f"  High failure risk (prob > 0.7): {np.sum(failure_probabilities > 0.7)}")
    print(f"  Mean prediction error: {rul_errors.mean():.2f} cycles")

# ============================================================================
# SECTION 9: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    # Load data
    dataset_number = 1  # Change to 1-4 for different datasets
    train_df, test_df, rul_df = load_cmapss_data(dataset_number)
    
    if train_df is None:
        return
    
    # Exploratory Data Analysis
    explore_data(train_df)
    
    # Prepare data for modeling
    (X_train, X_test, y_train_rul, y_test_rul, 
     y_train_failure, y_test_failure, feature_names, scaler) = prepare_data_for_modeling(
        train_df, test_df, rul_df
    )
    
    # Train RUL models
    rul_results, best_rul_model_name = train_rul_models(
        X_train, y_train_rul, X_test, y_test_rul
    )
    visualize_rul_results(rul_results, y_test_rul, best_rul_model_name)
    best_rul_model = rul_results[best_rul_model_name]['model']
    
    # Visualize feature importance for RUL
    visualize_feature_importance(best_rul_model, feature_names, task_name='RUL')
    
    # Train failure prediction models
    failure_results, best_failure_model_name = train_failure_models(
        X_train, y_train_failure, X_test, y_test_failure
    )
    visualize_failure_results(failure_results, y_test_failure, best_failure_model_name)
    best_failure_model = failure_results[best_failure_model_name]['model']
    
    # Visualize feature importance for failure prediction
    visualize_feature_importance(best_failure_model, feature_names, task_name='Failure')
    
    # Production monitoring simulation
    simulate_production_monitoring(
        best_rul_model, best_failure_model, X_test, 
        y_test_rul, y_test_failure, feature_names
    )
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - eda_visualization.png")
    print("  - rul_model_results.png")
    print("  - failure_model_results.png")
    print("  - feature_importance_rul.png")
    print("  - feature_importance_failure.png")
    print("  - production_monitoring_dashboard.png")

if __name__ == "__main__":
    main()