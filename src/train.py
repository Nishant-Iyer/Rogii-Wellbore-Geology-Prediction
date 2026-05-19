import pandas as pd
import numpy as np
import os
import glob
import pickle
import time
import argparse
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

from features import WellboreFeatureExtractor

def build_dataset(data_dir, num_wells=200, random_state=42):
    """
    Load a subset of training wells, extract features, and build train matrix.
    """
    extractor = WellboreFeatureExtractor(data_dir=data_dir)
    
    train_dir = os.path.join(data_dir, 'train')
    hw_paths = glob.glob(os.path.join(train_dir, '*__horizontal_well.csv'))
    
    # Sort and sample to ensure reproducibility
    hw_paths = sorted(hw_paths)
    if num_wells is not None and num_wells < len(hw_paths):
        np.random.seed(random_state)
        hw_paths = list(np.random.choice(hw_paths, size=num_wells, replace=False))
        
    print(f"Extracting features from {len(hw_paths)} wells...")
    start_time = time.time()
    
    dfs = []
    for i, hw_path in enumerate(hw_paths):
        well_id = os.path.basename(hw_path).split('__')[0]
        tw_path = os.path.join(train_dir, f"{well_id}__typewell.csv")
        
        if not os.path.exists(tw_path):
            continue
            
        df_hw = pd.read_csv(hw_path)
        df_tw = pd.read_csv(tw_path)
        
        # Extract features
        df_feat = extractor.extract_features(well_id, df_hw, df_tw, is_train=True)
        
        # We only train on rows where TVT is not NaN (which is all rows in training files,
        # but let's be safe and drop any NaNs in target)
        df_feat = df_feat.dropna(subset=['TVT', 'tvt_ref'])
        
        # Target variable is the residual
        df_feat['TVT_residual'] = df_feat['TVT'] - df_feat['tvt_ref']
        
        dfs.append(df_feat)
        
        if (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{len(hw_paths)} wells...")
            
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Dataset built in {time.time() - start_time:.2f}s. Total rows: {len(df_all)}")
    return df_all, extractor

def train_and_evaluate(data_dir, num_wells=200):
    """
    Perform GroupKFold CV, train LightGBM models, and save them.
    """
    df_all, extractor = build_dataset(data_dir, num_wells=num_wells)
    
    # Save the extractor (needed for inference to query spatial neighbors)
    os.makedirs('models', exist_ok=True)
    with open('models/extractor.pkl', 'wb') as f:
        pickle.dump(extractor, f)
        
    # Define features
    exclude_cols = ['MD', 'X', 'Y', 'Z', 'ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA', 
                    'TVT', 'TVT_input', 'TVT_residual', 'well_id', 'bias_damped', 'tvt_base']
    features = [c for c in df_all.columns if c not in exclude_cols]
    
    print(f"Number of training features: {len(features)}")
    print(f"Features: {features[:10]}... and {len(features)-10} more.")
    
    X = df_all[features].values
    y = df_all['TVT_residual'].values
    groups = df_all['well_id'].values
    
    gkf = GroupKFold(n_splits=5)
    
    oof_predictions = np.zeros(len(df_all))
    models = []
    
    # Track performance of baseline vs ML
    baseline_rmse_scores = []
    ml_rmse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Setup LightGBM
        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[]
        )
        
        # Predict on validation fold
        pred_res = model.predict(X_val)
        oof_predictions[val_idx] = pred_res
        models.append(model)
        
        # Calculate RMSE for this fold
        val_well_ids = groups[val_idx]
        val_df = df_all.iloc[val_idx]
        
        # Baseline: predicted residual is 0 (i.e. we use tvt_ref directly)
        baseline_pred = val_df['tvt_ref'].values
        baseline_rmse = np.sqrt(np.mean((val_df['TVT'].values - baseline_pred)**2))
        baseline_rmse_scores.append(baseline_rmse)
        
        # ML: baseline + predicted residual
        ml_pred = val_df['tvt_ref'].values + pred_res
        ml_rmse = np.sqrt(np.mean((val_df['TVT'].values - ml_pred)**2))
        ml_rmse_scores.append(ml_rmse)
        
        print(f"Fold {fold+1} - Baseline RMSE: {baseline_rmse:.4f} ft | ML RMSE: {ml_rmse:.4f} ft")
        
        # Save model
        with open(f'models/lgbm_fold_{fold}.pkl', 'wb') as f:
            pickle.dump(model, f)
            
    # Print overall performance
    overall_baseline_rmse = np.sqrt(np.mean((df_all['TVT'] - df_all['tvt_ref'])**2))
    overall_ml_rmse = np.sqrt(np.mean((df_all['TVT'] - (df_all['tvt_ref'] + oof_predictions))**2))
    
    print("\n" + "="*50)
    print("OVERALL CROSS-VALIDATION RESULTS")
    print("="*50)
    print(f"Baseline (Physics/Dip Model) RMSE: {overall_baseline_rmse:.4f} feet")
    print(f"ML (Baseline + LightGBM Residual) RMSE: {overall_ml_rmse:.4f} feet")
    print(f"Improvement: {overall_baseline_rmse - overall_ml_rmse:.4f} feet ({((overall_baseline_rmse - overall_ml_rmse)/overall_baseline_rmse)*100:.2f}%)")
    print("="*50)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_wells', type=int, default=150, help='Number of wells to train on')
    args = parser.parse_args()
    
    train_and_evaluate(data_dir='data', num_wells=args.num_wells)
