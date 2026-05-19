import pandas as pd
import numpy as np
import os
import glob
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d

FORMATIONS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']
MEAN_THICKNESSES = {
    'ANCC': -170.92,   # relative to ASTNU
    'ASTNU': -64.52,   # relative to ASTNL
    'ASTNL': 0.0,      # baseline anchor
    'EGFDU': 94.47,    # relative to ASTNL
    'EGFDL': 41.87,    # relative to EGFDU (so 136.34 relative to ASTNL)
    'BUDA': 124.75     # relative to EGFDL (so 261.09 relative to ASTNL)
}

def get_typewell_formation_tops(df_tw):
    """
    Extract the starting TVT (depth) for each geological formation in the typewell.
    Imputes missing formations using local layer thicknesses.
    """
    tops = {}
    for f in FORMATIONS:
        mask = df_tw['Geology'] == f
        if mask.any():
            tops[f] = df_tw.loc[mask, 'TVT'].min()
        else:
            tops[f] = np.nan
            
    # Impute missing values sequentially using anchors
    # We anchor everything relative to ASTNL (which is present in 100% of typewells)
    if pd.isna(tops['ASTNL']):
        # Fallback if ASTNL is somehow missing, find first non-nan
        for f in FORMATIONS:
            if not pd.isna(tops[f]):
                tops['ASTNL'] = tops[f] - (MEAN_THICKNESSES[f] - MEAN_THICKNESSES['ASTNL'])
                break
        if pd.isna(tops['ASTNL']):
            # Absolute fallback
            tops['ASTNL'] = 11600.0

    # Now propagate from ASTNL
    if pd.isna(tops['ASTNU']):
        tops['ASTNU'] = tops['ASTNL'] + MEAN_THICKNESSES['ASTNU']
    if pd.isna(tops['ANCC']):
        tops['ANCC'] = tops['ASTNU'] + MEAN_THICKNESSES['ANCC']
    if pd.isna(tops['EGFDU']):
        tops['EGFDU'] = tops['ASTNL'] + MEAN_THICKNESSES['EGFDU']
    if pd.isna(tops['EGFDL']):
        tops['EGFDL'] = tops['EGFDU'] + MEAN_THICKNESSES['EGFDL']
    if pd.isna(tops['BUDA']):
        tops['BUDA'] = tops['EGFDL'] + MEAN_THICKNESSES['BUDA']
        
    return tops

class WellboreFeatureExtractor:
    def __init__(self, data_dir, k_neighbors=5):
        self.data_dir = data_dir
        self.k_neighbors = k_neighbors
        self.train_wells_summary = None
        self.kdtree = None
        self._build_spatial_index()
        
    def _build_spatial_index(self):
        """
        Build a spatial KDTree of all training wells based on their average coordinates.
        """
        train_dir = os.path.join(self.data_dir, 'train')
        records = []
        for path in glob.glob(os.path.join(train_dir, '*__horizontal_well.csv')):
            well_id = os.path.basename(path).split('__')[0]
            # Use a fast preview to compute mean coordinate
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
            
    def get_neighbors(self, x_mean, y_mean, exclude_well_id=None):
        """
        Find nearest training well IDs to the target coordinates.
        """
        if self.kdtree is None:
            return []
            
        k = self.k_neighbors
        if exclude_well_id is not None:
            k += 1 # Query one extra since the well itself might be returned
            
        dists, indices = self.kdtree.query([x_mean, y_mean], k=k)
        if np.isscalar(dists):
            dists = [dists]
            indices = [indices]
            
        neighbor_ids = []
        for i in indices:
            nid = self.train_wells_summary.iloc[i]['well_id']
            if nid != exclude_well_id:
                neighbor_ids.append(nid)
                
        return neighbor_ids[:self.k_neighbors]

    def fit_dipping_planes(self, neighbor_ids):
        """
        Fit a 2D dipping plane (Z = w1 X + w2 Y + w0) for each formation top
        using the trajectories of neighboring training wells.
        """
        planes = {}
        neighbor_dfs = []
        for nid in neighbor_ids:
            path = os.path.join(self.data_dir, 'train', f'{nid}__horizontal_well.csv')
            if os.path.exists(path):
                neighbor_dfs.append(pd.read_csv(path))
                
        if not neighbor_dfs:
            return None
            
        df_combined = pd.concat(neighbor_dfs, ignore_index=True)
        
        for f in FORMATIONS:
            df_f = df_combined.dropna(subset=['X', 'Y', f])
            if len(df_f) > 10:
                lr = LinearRegression()
                lr.fit(df_f[['X', 'Y']].values, df_f[f].values)
                planes[f] = lr
            else:
                planes[f] = None
        return planes

    def extract_features(self, well_id, df_hw, df_tw, is_train=True):
        """
        Extract the complete feature set for a single horizontal well.
        """
        df = df_hw.copy()
        
        # 1. Trajectory differentials (drilling direction)
        df['dZ_dMD'] = df['Z'].diff().fillna(0.0)
        df['dX_dMD'] = df['X'].diff().fillna(0.0)
        df['dY_dMD'] = df['Y'].diff().fillna(0.0)
        
        # 2. Rolling GR statistics
        for w in [5, 10, 20, 50]:
            df[f'gr_roll_mean_{w}'] = df['GR'].rolling(window=w, min_periods=1).mean()
            df[f'gr_roll_std_{w}'] = df['GR'].rolling(window=w, min_periods=1).std().fillna(0.0)
            df[f'gr_roll_diff_{w}'] = df['GR'] - df[f'gr_roll_mean_{w}']
            
        # 3. Spatial Neighbors Dipping Planes baseline prediction
        # Find neighbors (excluding self to prevent validation leakage)
        x_mean = df['X'].mean()
        y_mean = df['Y'].mean()
        neighbor_ids = self.get_neighbors(x_mean, y_mean, exclude_well_id=well_id)
        
        planes = self.fit_dipping_planes(neighbor_ids)
        tw_tops = get_typewell_formation_tops(df_tw)
        
        # Calculate baseline TVT from neighbors' planes
        tvt_base_estimates = []
        for f in FORMATIONS:
            if planes is not None and planes[f] is not None:
                pred_f_z = planes[f].predict(df[['X', 'Y']].values)
                # TVT_est = TVT_top(F) - (Z - F_z_pred)
                tvt_est = tw_tops[f] - (df['Z'].values - pred_f_z)
                tvt_base_estimates.append(tvt_est)
                
        if tvt_base_estimates:
            df['tvt_base'] = np.mean(tvt_base_estimates, axis=0)
        else:
            # Fallback if no planes can be fit: make tvt_base = tw_tops['ASTNL'] - Z
            df['tvt_base'] = tw_tops['ASTNL'] - df['Z']
            
        # 4. 1D Damped Linear Bias Correction
        # Identify the known portion
        known_mask = df['TVT_input'].notna()
        
        if known_mask.sum() > 10:
            bias_known = df.loc[known_mask, 'TVT_input'] - df.loc[known_mask, 'tvt_base']
            mean_bias = bias_known.mean()
            
            # Fit 1D Linear trend along MD
            lr_1d = LinearRegression()
            lr_1d.fit(df.loc[known_mask, ['MD']].values, bias_known.values)
            
            # Predict linear bias correction
            bias_pred = lr_1d.predict(df[['MD']].values)
            
            # Apply Damped linear trend (transitioning to constant mean_bias after known zone)
            md_end_known = df.loc[known_mask, 'MD'].max()
            
            # Sigmoid decay damping factor: decay scale = 1000 feet
            decay_scale = 1000.0
            lambda_decay = 1.0 / (1.0 + np.exp((df['MD'].values - md_end_known) / decay_scale))
            
            df['bias_damped'] = lambda_decay * bias_pred + (1.0 - lambda_decay) * mean_bias
            df['tvt_ref'] = df['tvt_base'] + df['bias_damped']
        else:
            # If no known data (or too little), use default offset of 0
            df['tvt_ref'] = df['tvt_base']
            df['bias_damped'] = 0.0
            
        # 5. Typewell GR Lookup & Cross-Correlation features
        # Interpolate typewell GR log
        df_tw_clean = df_tw.dropna(subset=['TVT', 'GR'])
        tw_interp = interp1d(df_tw_clean['TVT'], df_tw_clean['GR'], bounds_error=False, fill_value='extrapolate')
        
        # Calculate GR mismatch features at various offsets
        offsets = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
        for offset in offsets:
            cand_tvt = df['tvt_ref'] + offset
            tw_gr_lookup = tw_interp(cand_tvt)
            
            df[f'tw_gr_offset_{offset}'] = tw_gr_lookup
            df[f'gr_diff_offset_{offset}'] = df['GR'] - tw_gr_lookup
            df[f'gr_sq_diff_offset_{offset}'] = (df['GR'] - tw_gr_lookup) ** 2
            
        # Add metadata features
        df['well_id'] = well_id
        
        return df
