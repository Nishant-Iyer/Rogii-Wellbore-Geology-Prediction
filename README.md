# ROGII Wellbore Geology Prediction

A geology-constrained pipeline for the **ROGII Wellbore Geology Prediction** competition on Kaggle. The task is to predict the True Vertical Thickness (TVT) along horizontal wellbores beyond the Prediction Start (PS) point, utilizing 3D XYZ trajectory coordinates and Gamma Ray (GR) log profiles.

**[Try the interactive demo](https://www.konnectingnots.com/demos/wellbore.html)**: the model runs in your browser, and you can change the known section, smoothness and search window. **[Case study](https://www.konnectingnots.com/projects/wellbore-geology.html)**

---

## 💡 Modeling Strategy

Instead of using unconstrained, high-variance regression models (such as LightGBM/XGBoost) which overfit to spatial coordinates and drift rapidly during lateral extrapolation, this pipeline uses a **hybrid physical dipping plane + geophysically-constrained dynamic programming (Viterbi)** model:

### 1. 2D Spatial Neighbor Dipping Plane
We use `scipy.spatial.cKDTree` to identify the 5 nearest neighboring wells in the training set and fit a 2D dipping plane:
$$Z = w_1 X + w_2 Y + w_0$$
This plane maps the regional structure of the target formation (`ASTNL`), allowing us to project structural trends into unknown lateral sections.

### 2. Typewell Entry Depth Back-Calculation
To handle cases where formation tops are missing in the test typewell logs, we estimate the formation top's TVT depth using the known portion of the target wellbore:
$$\text{TVT}_{\text{top}}(F) = \text{mean}_{x \in \text{known}} [\text{TVT}_{\text{input}}(x) + Z(x) - \hat{F}_z(x, y)]$$
This provides a highly stable anchor for the baseline TVT projection ($\text{TVT}_{\text{base}}$).

### 3. 1D Damped Linear Bias Correction
To correct minor local geological changes (such as minor faults or swelling layers) without introducing long-distance extrapolation drift, we fit a 1D linear trend to the bias along the Measured Depth (MD) on the known portion of the well. We apply a sigmoid damping function $\lambda(MD)$ that decays the linear correction back to the regional mean over a scale of 1000 feet:
$$\text{TVT}_{\text{ref}} = \text{TVT}_{\text{base}} + \lambda(MD) \cdot \text{bias}_{\text{pred}} + (1 - \lambda(MD)) \cdot \text{bias}_{\text{mean}}$$

### 4. Viterbi Dynamic Programming Path Finder
To match local geological layers, we frame the alignment between the horizontal wellbore's GR log and the vertical typewell's GR profile as a sequence optimization problem:
- **States:** A fine grid of TVT offsets from the baseline projection (from $-30$ to $+30$ feet with a step of $0.5$ feet).
- **Data Cost:** The squared difference between the horizontal wellbore's GR measurement and the interpolated typewell's GR value at the state depth.
- **Transition Cost:** A quadratic penalty on the difference between consecutive states, ensuring geological smoothness and preventing sudden non-physical jumps:
  $$C_{\text{trans}} = w_{\text{smooth}} \cdot (\text{offset}_i - \text{offset}_{i-1})^2$$
Using the Viterbi algorithm, we compute the globally optimal path through all depth offsets.

## 📊 Results

Evaluated with `src/evaluate.py` on the three public sample wells. For each well the depth after the known section is hidden, and the regional plane is fitted only from the *other* wells so the well's own answers can't leak in.

| Well | Points predicted | Baseline only | With path matching |
|---|---|---|---|
| 000d7d20 | 3,836 | 19.5 ft | **10.1 ft** |
| 00bbac68 | 6,014 | 21.4 ft | **16.1 ft** |
| 00e12e8b | 4,301 | 17.1 ft | **7.2 ft** |
| **All** | 14,151 | 19.6 ft | **12.4 ft** |

Path matching cuts the error by about 37% over the regional-plane baseline.

> **Note on the public leaderboard:** the three public test wells also appear in the training data, so copying their known values scores a perfect 0. `predict.py` can still do this with `--use-train-match`, but it's off by default because it measures nothing about the model.

---

## 📁 Project Structure

```
├── src/
│   ├── download_data.py   # Ingests and downloads competition data via Kaggle API
│   ├── features.py        # Feature engineering, spatial indexing, dipping planes
│   ├── train.py           # Cross-validation and GBDT residual model trainer
│   ├── predict.py         # Main prediction pipeline (plane + bias correction + Viterbi)
│   └── evaluate.py        # Leave-one-out evaluation on wells with known answers
├── requirements.txt       # Python package dependencies
├── .gitignore             # Git ignore patterns
└── README.md              # Documentation
```

---

## 🚀 Setup & Execution

### 1. Prerequisites
- Python 3.11+
- Kaggle API token configured (either in `~/.kaggle/kaggle.json` or exported via `KAGGLE_USERNAME` and `KAGGLE_KEY`).

### 2. Install Dependencies
Create a virtual environment and install the required libraries:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download Competition Data
To download the dataset, run:
```bash
python src/download_data.py
```

### 4. Run Prediction Pipeline
To generate the final predictions and run the validation suite:
```bash
python src/predict.py
```
This will output `submission.csv` in the root folder, ready for submission!

To reproduce the results table on the sample wells:
```bash
python src/evaluate.py --data-dir sample_data
```

### 5. Launch the Streamlit Dashboard
To launch the interactive, high-fidelity visualization dashboard locally:
```bash
streamlit run app.py
```
This starts a local development server at `http://localhost:8501` featuring real-time parameter tuning, 3D wellbore trajectory plotting, and Viterbi alignment simulations.
