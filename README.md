# NASA Turbofan Engine Predictive Maintenance Project

## 🎯 Project Overview

This project implements end-to-end machine learning for predictive maintenance of aircraft turbofan engines using the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset.

**Goals:**

- Predict Remaining Useful Life (RUL) of engines with MAE < 15 cycles
- Classify engines at risk of failure within 30 cycles (Recall > 80%)
- Demonstrate production-ready ML system design

---

## 📊 Dataset Information

### NASA C-MAPSS Dataset

- **Source:** [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) or [NASA Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- **Type:** Turbofan engine degradation simulation
- **Size:** 4 datasets (FD001-FD004) with varying complexity
- **Features:** 21 sensor measurements + 3 operational settings
- **Format:** Space-separated text files

### Dataset Variants

| Dataset | Operating Conditions | Fault Modes | Complexity |
|---------|---------------------|-------------|------------|
| FD001   | 1                   | 1           | Simple     |
| FD002   | 6                   | 1           | Medium     |
| FD003   | 1                   | 2           | Medium     |
| FD004   | 6                   | 2           | Complex    |

**Recommendation:** Start with FD001 for learning, then progress to FD004 for challenge.

---

## 🚀 Quick Start

### Step 1: Download the Dataset

**Option A: Kaggle (Easiest)**

1. Create free Kaggle account
2. Visit: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
3. Click "Download" button
4. Extract ZIP file

**Option B: NASA Direct**

1. Visit: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
2. Find "Turbofan Engine Degradation Simulation"
3. Download ZIP file
4. Extract files

### Step 2: Setup Environment

```bash
# Create virtual environment (recommended)
python -m venv turbofan_env
source turbofan_env/bin/activate  # On Windows: turbofan_env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Step 3: Organize Files

Place downloaded dataset files in your project directory:

```
your_project_folder/
│
├── v1.py                          # Main script
├── train_FD001.txt                # Training data
├── test_FD001.txt                 # Test data
├── RUL_FD001.txt                  # True RUL values
├── train_FD002.txt                # (Optional: other datasets)
├── test_FD002.txt
├── ...
└── README.md
```

### Step 4: Run the Pipeline

```bash
# Run complete pipeline
python v1.py
```

**Expected Runtime:** 5-15 minutes depending on your machine

---

## 📈 What the Script Does

### Pipeline Stages

1. **Data Loading & Validation**
   - Loads train/test datasets
   - Validates file structure
   - Reports basic statistics

2. **Exploratory Data Analysis**
   - Engine lifecycle distributions
   - Sensor degradation patterns
   - Correlation analysis
   - Operating conditions visualization
   - **Output:** `eda_visualization.png`

3. **Feature Engineering**
   - RUL target creation
   - Rolling statistics (5, 10-cycle windows)
   - Degradation features (diff from start, normalized)
   - Binary classification target (failure within 30 cycles)
   - **Result:** ~200+ engineered features

4. **RUL Prediction (Regression)**
   - Trains Random Forest & Gradient Boosting models
   - Evaluates: MAE, RMSE, R²
   - Selects best performer
   - **Output:** `rul_model_results.png`

5. **Failure Prediction (Classification)**
   - Trains binary classifiers
   - Evaluates: Precision, Recall, F1, ROC-AUC
   - Confusion matrix & ROC curve
   - **Output:** `failure_model_results.png`

6. **Feature Importance Analysis**
   - Identifies top predictive features
   - Visualizes importance rankings
   - **Outputs:** `feature_importance_rul.png`, `feature_importance_failure.png`

7. **Production Monitoring Simulation**
   - Simulates deployment monitoring
   - Error tracking & alerting
   - Drift detection framework
   - **Output:** `production_monitoring_dashboard.png`

---

## 📊 Expected Results

### Target Performance Metrics

| Metric | Target | Expected (FD001) |
|--------|--------|------------------|
| RUL MAE | < 15 cycles | 10-14 cycles |
| RUL RMSE | < 20 cycles | 13-18 cycles |
| Classification Recall | > 80% | 75-85% |
| Classification Precision | > 60% | 65-75% |
| F1 Score | > 0.70 | 0.70-0.80 |

### Generated Visualizations

After running the script, you'll have 6 PNG files:

1. **eda_visualization.png** - Data exploration dashboard
2. **rul_model_results.png** - RUL prediction performance
3. **failure_model_results.png** - Classification results
4. **feature_importance_rul.png** - Top features for RUL
5. **feature_importance_failure.png** - Top features for classification
6. **production_monitoring_dashboard.png** - Deployment simulation

---

## 🔧 Customization Options

### Try Different Datasets

```python
# In main() function, change dataset number:
dataset_number = 2  # FD002
# Options: 1 (FD001), 2 (FD002), 3 (FD003), 4 (FD004)
```

### Adjust Feature Engineering

```python
# Modify rolling window sizes:
df = add_rolling_features(df, window_sizes=[3, 5, 10, 20])

# Change failure prediction window:
df['failure_within_30'] = (df['RUL'] <= 20).astype(int)  # 20 cycles instead of 30
```

### Experiment with Models

```python
# Add more models in train_rul_model():
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42)
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_
```

---

## 🎓 Learning Path

### Beginner

1. Run script with FD001 dataset
2. Review all generated visualizations
3. Understand feature engineering logic
4. Interpret model performance metrics

### Intermediate

1. Try all 4 datasets (FD001-FD004)
2. Modify feature engineering parameters
3. Add new models (XGBoost, LightGBM)
4. Implement cross-validation
5. Tune hyperparameters

### Advanced

1. Implement LSTM/GRU for sequence modeling
2. Add ensemble methods (stacking, blending)
3. Implement real-time prediction API
4. Build web dashboard with Flask/Streamlit
5. Deploy to cloud (AWS, Azure, GCP)

---

## 🧪 Next Steps & Extensions

### Short-term Improvements

- [ ] Implement time-series cross-validation
- [ ] Add SHAP values for explainability
- [ ] Create interactive Plotly dashboards
- [ ] Save trained models (pickle/joblib)
- [ ] Add unit tests

### Advanced Features

- [ ] LSTM/GRU neural networks for sequences
- [ ] Anomaly detection for drift monitoring
- [ ] Multi-task learning (RUL + failure classification)
- [ ] Transfer learning across datasets
- [ ] Real-time streaming pipeline

### Production Deployment

- [ ] Build REST API (FastAPI/Flask)
- [ ] Create monitoring dashboard (Streamlit)
- [ ] Implement CI/CD pipeline
- [ ] Add logging and error handling
- [ ] Deploy to cloud platform
- [ ] Setup automated retraining

---

## 📚 Key Concepts Demonstrated

### ML System Design Decisions

✓ **Cloud vs Edge:** Hybrid architecture (edge collection, cloud training)  
✓ **Offline vs Online Learning:** Offline batch retraining (quarterly)  
✓ **Batch vs Online Prediction:** Daily batch predictions  

### Production Considerations

✓ Data drift monitoring  
✓ Concept drift (feedback loops)  
✓ Training-serving skew prevention  
✓ Class imbalance handling  
✓ Feature importance analysis  
✓ Model performance tracking  

### CRISP-DM Framework

✓ Business Understanding (problem definition)  
✓ Data Understanding (EDA)  
✓ Data Preparation (feature engineering)  
✓ Modeling (multiple algorithms)  
✓ Evaluation (comprehensive metrics)  
✓ Deployment (monitoring simulation)  

---

## 🐛 Troubleshooting

### "FileNotFoundError"

**Problem:** Cannot find dataset files  
**Solution:** 

1. Verify files are in same directory as script
2. Check file names match exactly: `train_FD001.txt` (case-sensitive)
3. Extract ZIP completely

### "Memory Error"

**Problem:** Not enough RAM for full dataset  
**Solution:**

1. Use FD001 (smallest dataset)
2. Reduce rolling window sizes
3. Sample fewer engines for training

### "Module not found"

**Problem:** Missing Python packages  
**Solution:**

```bash
pip install -r requirements.txt
```

### Poor Model Performance

**Problem:** MAE > 20 cycles or Recall < 70%  
**Solution:**

1. Check data loading (no errors?)
2. Verify feature engineering ran correctly
3. Try different model (Gradient Boosting often works better)
4. Increase training data (use more engines)
5. Try different dataset (FD001 is easiest)

---

## 📖 References

### Academic Papers

1. Saxena, A., & Goebel, K. (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository
2. Zheng, S., et al. (2017). "Long Short-Term Memory Network for Remaining Useful Life estimation"
3. Li, X., et al. (2018). "Remaining useful life estimation in prognostics using deep convolution neural networks"

### Useful Resources

- [NASA Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- [Kaggle Notebooks](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps/code)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Predictive Maintenance Review Paper](https://www.mdpi.com/2076-3417/10/23/8548)

---

## 🤝 Contributing

This is an educational project. Feel free to:

- Experiment with different approaches
- Share your results and improvements
- Ask questions and provide feedback

---

## 📝 License

This project uses publicly available NASA data. The code is provided for educational purposes.

---

## 💡 Tips for Success

1. **Start Simple:** Run with default settings first
2. **Understand Metrics:** MAE tells you average prediction error in cycles
3. **Check Visualizations:** They reveal insights about data and model behavior
4. **Iterate Gradually:** Change one thing at a time
5. **Document Changes:** Keep notes on what works and what doesn't
6. **Be Patient:** First run takes longer (data processing)
7. **Compare Results:** Try all 4 datasets to see how complexity affects performance

---

**Good luck with your predictive maintenance project! 🚀**

## written by Dr. Ehsan Zafari 
