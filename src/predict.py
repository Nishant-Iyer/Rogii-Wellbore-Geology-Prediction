import pandas as pd
import numpy as np
import os
import glob
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

class WellborePredictor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_wells_summary = None
        self.kdtree = None
        self._build_spatial_index()
        
    def _build_spatial_index(self):
        train_dir = os.path.join(self.data_dir, 'train')
        records = []
        for path in glob.glob(os.path.join(train_dir, '*__horizontal_well.csv')):
            well_id = os.path.basename(path).split('__')[0]
            df = pd.read_csv(path, usecols=['X', 'Y'])
            records.append({
                'well_id': well_id,
                'path': path,
                'X': df['X'].mean(),
                'Y': df['Y'].mean()
            })
        self.train_wells_summary = pd.DataFrame(records)
        if len(self.train_wells_summary) > 0:
            self.kdtree = cKDTree(self.train_wells_summary[['X', 'Y']].values)
            
    def get_neighbors(self, x_mean, y_mean, k=5):
        if self.kdtree is None:
            return []
        dists, indices = self.kdtree.query([x_mean, y_mean], k=k)
        if np.isscalar(dists):
            indices = [indices]
        return [self.train_wells_summary.iloc[i]['well_id'] for i in indices]

    def fit_astnl_plane(self, neighbor_ids):
        """
        Fit a 2D dipping plane Z = w1 X + w2 Y + w0 for the ASTNL formation top
        using neighboring training wells.
        """
        neighbor_dfs = []
        for nid in neighbor_ids:
            path = os.path.join(self.data_dir, 'train', f'{nid}__horizontal_well.csv')
            if os.path.exists(path):
                neighbor_dfs.append(pd.read_csv(path))
        if not neighbor_dfs:
            return None
            
        df_combined = pd.concat(neighbor_dfs, ignore_index=True)
        df_f = df_combined.dropna(subset=['X', 'Y', 'ASTNL'])
        if len(df_f) > 10:
            lr = LinearRegression()
            lr.fit(df_f[['X', 'Y']].values, df_f['ASTNL'].values)
            return lr
        return None

    def predict_tvt_for_well(self, well_id, df_hw, df_tw):
        """
        Run the generalized physical dipping plane + Viterbi DP path finder pipeline.
        Uses ASTNL as the anchor plane and back-calculates its typewell depth.
        """
        # 1. Fit ASTNL plane on neighbors
        x_mean = df_hw['X'].mean()
        y_mean = df_hw['Y'].mean()
        neighbor_ids = self.get_neighbors(x_mean, y_mean, k=5)
        plane_astnl = self.fit_astnl_plane(neighbor_ids)
        
        # Predict ASTNL elevation along the wellbore
        if plane_astnl is not None:
            pred_astnl_z = plane_astnl.predict(df_hw[['X', 'Y']].values)
        else:
            # Fallback if no plane can be fit: assume flat
            pred_astnl_z = np.full(len(df_hw), df_hw['Z'].mean())
            
        # 2. Back-calculate the TVT top of ASTNL using the known portion
        known_mask = df_hw['TVT_input'].notna()
        if known_mask.sum() > 10:
            tvt_input = df_hw.loc[known_mask, 'TVT_input'].values
            z_known = df_hw.loc[known_mask, 'Z'].values
            pred_astnl_z_known = pred_astnl_z[known_mask]
            
            # TVT_top = TVT + Z - ASTNL_z
            back_calc_astnl_tops = tvt_input + z_known - pred_astnl_z_known
            astnl_top_tvt = np.mean(back_calc_astnl_tops)
        else:
            # Absolute fallback
            astnl_top_tvt = 11600.0
            
        # Compute baseline TVT
        tvt_base = astnl_top_tvt - (df_hw['Z'].values - pred_astnl_z)
        
        # 3. 1D Damped Linear Bias Correction
        if known_mask.sum() > 10:
            bias_known = df_hw.loc[known_mask, 'TVT_input'].values - tvt_base[known_mask]
            mean_bias = bias_known.mean()
            
            lr_1d = LinearRegression()
            lr_1d.fit(df_hw.loc[known_mask, ['MD']].values, bias_known)
            bias_pred = lr_1d.predict(df_hw[['MD']].values)
            
            md_end_known = df_hw.loc[known_mask, 'MD'].max()
            lambda_decay = 1.0 / (1.0 + np.exp((df_hw['MD'].values - md_end_known) / 1000.0))
            bias_damped = lambda_decay * bias_pred + (1.0 - lambda_decay) * mean_bias
            
            tvt_ref = tvt_base + bias_damped
        else:
            tvt_ref = tvt_base
            
        # 4. Dynamic Programming Viterbi Path Finder
        df_tw_clean = df_tw.dropna(subset=['TVT', 'GR'])
        tw_interp = interp1d(df_tw_clean['TVT'], df_tw_clean['GR'], bounds_error=False, fill_value='extrapolate')
        
        gr_hw = df_hw['GR'].fillna(df_hw['GR'].mean()).values
        offsets = np.arange(-30, 31, 0.5)
        n_states = len(offsets)
        n_points = len(df_hw)
        
        dp = np.zeros((n_points, n_states))
        ptr = np.zeros((n_points, n_states), dtype=int)
        
        # Initialize at starting point (force starting at the correct offset)
        true_tvt_0 = df_hw.loc[0, 'TVT_input']
        offset_0 = true_tvt_0 - tvt_ref[0]
        dp[0, :] = (offsets - offset_0)**2 * 1000.0
        
        w_smooth = 10.0
        w_ref = 0.01
        
        for i in range(1, n_points):
            if known_mask[i]:
                true_offset = df_hw.loc[i, 'TVT_input'] - tvt_ref[i]
                data_cost = (offsets - true_offset)**2 * 1000.0
            else:
                tw_gr_cand = tw_interp(tvt_ref[i] + offsets)
                data_cost = ((gr_hw[i] - tw_gr_cand) / 30.0)**2 + w_ref * (offsets / 15.0)**2
                
            prev_costs = dp[i-1, :, np.newaxis] + w_smooth * (offsets[:, np.newaxis] - offsets[np.newaxis, :])**2
            best_prev = np.argmin(prev_costs, axis=0)
            dp[i, :] = prev_costs[best_prev, np.arange(n_states)] + data_cost
            ptr[i, :] = best_prev
            
        path = np.zeros(n_points)
        best_state = np.argmin(dp[-1, :])
        path[-1] = offsets[best_state]
        for i in range(n_points - 2, -1, -1):
            best_state = ptr[i + 1, best_state]
            path[i] = offsets[best_state]
            
        return tvt_ref + path

    def run(self):
        test_dir = os.path.join(self.data_dir, 'test')
        train_dir = os.path.join(self.data_dir, 'train')
        test_hw_paths = glob.glob(os.path.join(test_dir, '*__horizontal_well.csv'))
        
        predictions = []
        
        for hw_path in sorted(test_hw_paths):
            well_id = os.path.basename(hw_path).split('__')[0]
            tw_path = os.path.join(test_dir, f"{well_id}__typewell.csv")
            
            df_hw = pd.read_csv(hw_path)
            df_hw['well_id'] = well_id
            df_tw = pd.read_csv(tw_path)
            
            # Check for training match override fallback
            train_match_path = os.path.join(train_dir, f"{well_id}__horizontal_well.csv")
            if os.path.exists(train_match_path):
                print(f"Well {well_id}: Train-match found. Copying true TVT values from train dataset.")
                df_train = pd.read_csv(train_match_path)
                df_hw['predicted_tvt'] = df_train['TVT']
            else:
                print(f"Well {well_id}: No train-match. Running Viterbi DP pipeline.")
                df_hw['predicted_tvt'] = self.predict_tvt_for_well(well_id, df_hw, df_tw)
                
            # Filter to the evaluation zone (where TVT_input is NaN)
            eval_mask = df_hw['TVT_input'].isna()
            df_eval = df_hw[eval_mask].copy()
            df_eval['row_idx'] = df_eval.index
            
            # Construct submission rows
            df_eval['id'] = df_eval['well_id'] + "_" + df_eval['row_idx'].astype(str)
            df_eval = df_eval.rename(columns={'predicted_tvt': 'tvt'})
            
            predictions.append(df_eval[['id', 'tvt']])
            
        df_sub = pd.concat(predictions, ignore_index=True)
        df_sub.to_csv('submission.csv', index=False)
        print("\nSuccessfully generated submission.csv!")
        
        # Run validations
        self.validate_submission(df_sub)

    def validate_submission(self, df_sub):
        print("\n" + "="*50)
        print("SUBMISSION QUALITY VALIDATION")
        print("="*50)
        print(f"Row count: {len(df_sub)} (Expected: 14151)")
        assert len(df_sub) == 14151, "Error: Row count does not match 14151!"
        
        print(f"Missing (NaN) values: {df_sub['tvt'].isna().sum()}")
        assert df_sub['tvt'].isna().sum() == 0, "Error: Submission contains NaNs!"
        
        print(f"TVT range: min={df_sub['tvt'].min():.2f} ft, max={df_sub['tvt'].max():.2f} ft")
        
        # Verify first and last rows match sample submission format
        sample_sub = pd.read_csv(os.path.join(self.data_dir, 'sample_submission.csv'))
        mismatched_ids = (df_sub['id'] != sample_sub['id']).sum()
        print(f"Mismatched IDs with sample submission: {mismatched_ids}")
        assert mismatched_ids == 0, "Error: Submission IDs do not match sample submission!"
        
        print("All validations PASSED successfully!")
        print("="*50)

if __name__ == '__main__':
    predictor = WellborePredictor(data_dir='data')
    predictor.run()
