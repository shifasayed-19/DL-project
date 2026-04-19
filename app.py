"""
Flask backend — Explainable Credit Scoring System
DNN + WOE + SHAP · UCI German Credit Dataset
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import torch
import torch.nn as nn
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Device ────────────────────────────────────────────────────────────────
device = torch.device('cpu')

# ── Global state ──────────────────────────────────────────────────────────
model          = None
model_metadata = None
THRESHOLD      = 0.50
NUM_BINS       = 5


# ══════════════════════════════════════════════════════════════════════════
# Model architecture — must exactly match notebook Cell 12
# ══════════════════════════════════════════════════════════════════════════
class CreditDNN(nn.Module):
    def __init__(self, input_dim, dropout=0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════
# Load checkpoint saved by notebook Cell 14
# ══════════════════════════════════════════════════════════════════════════
def load_model(model_path: str):
    global model, model_metadata

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Pull everything the notebook saved into the checkpoint
    model_metadata = {
        'selected_features': checkpoint['selected_features'],
        'woe_maps':          checkpoint['woe_maps'],          # {feat: {bin: woe}}
        'scaler_mean':       np.array(checkpoint['scaler_mean'],  dtype=np.float32),
        'scaler_scale':      np.array(checkpoint['scaler_scale'], dtype=np.float32),
        'threshold':         float(checkpoint.get('threshold', THRESHOLD)),
        'input_dim':         int(checkpoint['input_dim']),
    }

    model = CreditDNN(model_metadata['input_dim']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Model loaded — {model_metadata['input_dim']} features  "
          f"| threshold={model_metadata['threshold']}")
    print(f"   Selected features: {model_metadata['selected_features']}")


# ══════════════════════════════════════════════════════════════════════════
# Preprocessing — mirrors the notebook pipeline exactly:
#   raw value → quantile bin (0-4) → WOE score → StandardScaler
#
# The notebook's KBinsDiscretizer uses 'quantile' strategy on the
# TRAINING data.  We replicate binning with the bin-edge boundaries
# stored in the checkpoint (if present) or fall back to equal-width
# approximation so the pipeline still runs without retraining.
# ══════════════════════════════════════════════════════════════════════════

# Raw feature ranges from the German Credit dataset (used for binning
# when bin edges are not stored in the checkpoint)
FEATURE_RANGES = {
    'Attribute1':  (0,  3),   # checking account status
    'Attribute2':  (4,  72),  # duration months
    'Attribute3':  (0,  4),   # credit history
    'Attribute4':  (0,  10),  # purpose
    'Attribute5':  (250, 18420), # credit amount
    'Attribute6':  (0,  4),   # savings
    'Attribute7':  (0,  4),   # employment
    'Attribute8':  (1,  4),   # instalment rate
    'Attribute9':  (0,  1),   # personal status
    'Attribute10': (0,  1),   # other debtors
    'Attribute11': (0,  4),   # present residence
    'Attribute12': (0,  10),  # property
    'Attribute13': (19, 75),  # age
    'Attribute14': (0,  2),   # other installment plans
    'Attribute15': (0,  1),   # housing
    'Attribute16': (1,  4),   # existing credits
    'Attribute17': (0,  4),   # job
    'Attribute18': (1,  4),   # liable people
    'Attribute19': (0,  1),   # telephone
    'Attribute20': (0,  1),   # foreign worker
}


def _raw_to_bin(value: float, feat: str, bin_edges: list = None) -> int:
    """
    Convert a raw feature value to its quantile bin index (0 to NUM_BINS-1).
    Uses stored bin edges if available, otherwise equal-width approximation.
    """
    if bin_edges is not None:
        # bin_edges: sorted list of NUM_BINS-1 thresholds
        for i, edge in enumerate(bin_edges):
            if value <= edge:
                return i
        return NUM_BINS - 1

    # Fallback: equal-width bins over known feature range
    lo, hi = FEATURE_RANGES.get(feat, (0, 1))
    if hi == lo:
        return 0
    frac = (value - lo) / (hi - lo)
    return int(np.clip(int(frac * NUM_BINS), 0, NUM_BINS - 1))


def preprocess_input(raw_values: dict) -> np.ndarray:
    """
    Full preprocessing pipeline matching the notebook:
      1. For each selected feature: raw → bin → WOE
      2. Stack into vector
      3. StandardScaler transform (using saved mean/scale)
    Returns numpy array shape (1, n_selected_features)
    """
    selected_features = model_metadata['selected_features']
    woe_maps          = model_metadata['woe_maps']
    scaler_mean       = model_metadata['scaler_mean']
    scaler_scale      = model_metadata['scaler_scale']
    # Optional: bin edges saved from KBinsDiscretizer (if notebook was updated to save them)
    bin_edges_map     = model_metadata.get('bin_edges', {})

    feature_vector = []
    for feat in selected_features:
        raw_val    = float(raw_values.get(feat, 0.0))
        bin_edges  = bin_edges_map.get(feat, None)
        bin_idx    = _raw_to_bin(raw_val, feat, bin_edges)

        # woe_maps[feat] keys may be int or float — try both
        woe_map = woe_maps.get(feat, {})
        woe_val = woe_map.get(bin_idx,
                  woe_map.get(float(bin_idx), 0.0))
        feature_vector.append(woe_val)

    arr = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    arr = (arr - scaler_mean) / scaler_scale          # StandardScaler
    return arr


# ══════════════════════════════════════════════════════════════════════════
# Feature metadata for the UI form
# ══════════════════════════════════════════════════════════════════════════
FEATURE_DEFINITIONS = {
    "Attribute1": {
        "label": "Checking Account Status",
        "description": "Status of existing checking account",
        "type": "select",
        "options": [
            {"value": 0, "label": "No checking account"},
            {"value": 1, "label": "Negative balance (< 0 DM)"},
            {"value": 2, "label": "Low balance (0–200 DM)"},
            {"value": 3, "label": "Healthy balance (≥ 200 DM)"},
        ]
    },
    "Attribute2": {
        "label": "Loan Duration (months)",
        "description": "Duration of the credit in months",
        "type": "range", "min": 4, "max": 72, "step": 1, "default": 18
    },
    "Attribute3": {
        "label": "Credit History",
        "description": "Past credit repayment behaviour",
        "type": "select",
        "options": [
            {"value": 0, "label": "No credits taken / all paid duly"},
            {"value": 1, "label": "All credits paid back duly"},
            {"value": 2, "label": "Existing credits paid duly"},
            {"value": 3, "label": "Delay in paying off in the past"},
            {"value": 4, "label": "Critical / other credits existing"},
        ]
    },
    "Attribute4": {
        "label": "Loan Purpose",
        "description": "Purpose of the credit request",
        "type": "select",
        "options": [
            {"value": 0,  "label": "Car (new)"},
            {"value": 1,  "label": "Car (used)"},
            {"value": 2,  "label": "Furniture / equipment"},
            {"value": 3,  "label": "Radio / TV"},
            {"value": 4,  "label": "Domestic appliances"},
            {"value": 5,  "label": "Repairs"},
            {"value": 6,  "label": "Education"},
            {"value": 7,  "label": "Vacation"},
            {"value": 8,  "label": "Retraining"},
            {"value": 9,  "label": "Business"},
            {"value": 10, "label": "Other"},
        ]
    },
    "Attribute5": {
        "label": "Credit Amount (DM)",
        "description": "Total amount of credit requested",
        "type": "range", "min": 250, "max": 18420, "step": 50, "default": 2500
    },
    "Attribute6": {
        "label": "Savings / Bonds",
        "description": "Balance in savings account or bonds",
        "type": "select",
        "options": [
            {"value": 0, "label": "Unknown / no savings account"},
            {"value": 1, "label": "< 100 DM"},
            {"value": 2, "label": "100–500 DM"},
            {"value": 3, "label": "500–1000 DM"},
            {"value": 4, "label": "≥ 1000 DM"},
        ]
    },
    "Attribute7": {
        "label": "Employment Duration",
        "description": "Years at current employer",
        "type": "select",
        "options": [
            {"value": 0, "label": "Unemployed"},
            {"value": 1, "label": "< 1 year"},
            {"value": 2, "label": "1–4 years"},
            {"value": 3, "label": "4–7 years"},
            {"value": 4, "label": "≥ 7 years"},
        ]
    },
    "Attribute8": {
        "label": "Instalment Rate (% of income)",
        "description": "Monthly instalment as % of disposable income",
        "type": "range", "min": 1, "max": 4, "step": 1, "default": 2
    },
    "Attribute13": {
        "label": "Age (years)",
        "description": "Applicant age in years",
        "type": "range", "min": 19, "max": 75, "step": 1, "default": 35
    },
    "Attribute20": {
        "label": "Foreign Worker",
        "description": "Is the applicant a foreign worker?",
        "type": "select",
        "options": [
            {"value": 0, "label": "No"},
            {"value": 1, "label": "Yes"},
        ]
    },
}


# ══════════════════════════════════════════════════════════════════════════
# Explanation helper — rule-based, same logic regardless of model mode
# ══════════════════════════════════════════════════════════════════════════
def _generate_explanations(data: dict, prob: float) -> list:
    factors = []

    acct = int(data.get('Attribute1', 2))
    acct_labels = {0: "no checking account", 1: "negative balance",
                   2: "low balance (0–200 DM)", 3: "healthy balance"}
    if acct <= 1:
        factors.append({"feature": "Checking Account", "direction": "risk",
                        "explanation": f"Checking account shows {acct_labels[acct]}, a strong default risk signal.",
                        "impact": "high"})
    elif acct == 3:
        factors.append({"feature": "Checking Account", "direction": "protect",
                        "explanation": "Healthy checking account balance significantly lowers default risk.",
                        "impact": "high"})

    hist = int(data.get('Attribute3', 2))
    if hist >= 3:
        factors.append({"feature": "Credit History", "direction": "risk",
                        "explanation": "Past delays or critical accounts in credit history raise risk substantially.",
                        "impact": "high"})
    elif hist <= 1:
        factors.append({"feature": "Credit History", "direction": "protect",
                        "explanation": "Clean credit history — all past credits paid duly.",
                        "impact": "medium"})

    dur = int(data.get('Attribute2', 18))
    if dur > 36:
        factors.append({"feature": "Loan Duration", "direction": "risk",
                        "explanation": f"Long repayment period ({dur} months) increases default exposure.",
                        "impact": "medium"})
    elif dur <= 12:
        factors.append({"feature": "Loan Duration", "direction": "protect",
                        "explanation": f"Short loan duration ({dur} months) reduces default risk.",
                        "impact": "low"})

    amt = float(data.get('Attribute5', 2500))
    if amt > 8000:
        factors.append({"feature": "Credit Amount", "direction": "risk",
                        "explanation": f"High loan amount (DM {amt:,.0f}) increases financial burden.",
                        "impact": "medium"})

    sav = int(data.get('Attribute6', 1))
    if sav >= 3:
        factors.append({"feature": "Savings / Bonds", "direction": "protect",
                        "explanation": "Substantial savings demonstrate financial stability.",
                        "impact": "medium"})
    elif sav == 0:
        factors.append({"feature": "Savings / Bonds", "direction": "risk",
                        "explanation": "No savings account or bonds — limited financial cushion.",
                        "impact": "medium"})

    emp = int(data.get('Attribute7', 2))
    if emp >= 3:
        factors.append({"feature": "Employment", "direction": "protect",
                        "explanation": "Long-term stable employment is a strong positive indicator.",
                        "impact": "medium"})
    elif emp == 0:
        factors.append({"feature": "Employment", "direction": "risk",
                        "explanation": "Unemployment is a major default risk factor.",
                        "impact": "high"})

    age = int(data.get('Attribute13', 35))
    if age < 25:
        factors.append({"feature": "Age", "direction": "risk",
                        "explanation": f"Young applicant ({age} yrs) — statistically higher default rate.",
                        "impact": "low"})
    elif age >= 45:
        factors.append({"feature": "Age", "direction": "protect",
                        "explanation": f"Mature applicant ({age} yrs) — statistically lower default rate.",
                        "impact": "low"})

    return factors


# ══════════════════════════════════════════════════════════════════════════
# Demo / fallback prediction (heuristic, no model needed)
# ══════════════════════════════════════════════════════════════════════════
def _demo_predict(data: dict):
    prob = 0.30
    acct = int(data.get('Attribute1', 2))
    if acct == 0:   prob += 0.15
    elif acct == 1: prob += 0.20
    elif acct == 2: prob += 0.05

    hist = int(data.get('Attribute3', 2))
    if hist == 3:   prob += 0.10
    elif hist == 4: prob += 0.15

    dur = int(data.get('Attribute2', 18))
    prob += min((dur - 12) / 120, 0.15)

    amt = float(data.get('Attribute5', 2500))
    prob += min((amt - 1000) / 50000, 0.12)

    sav = int(data.get('Attribute6', 1))
    prob -= sav * 0.04

    emp = int(data.get('Attribute7', 2))
    prob -= emp * 0.03

    age = int(data.get('Attribute13', 35))
    if age < 25:  prob += 0.05
    elif age > 50: prob -= 0.03

    prob = float(np.clip(prob, 0.03, 0.97))
    decision = 'DENIED' if prob >= THRESHOLD else 'APPROVED'
    risk_level = (
        'Very High' if prob >= 0.80 else
        'High'      if prob >= 0.60 else
        'Medium'    if prob >= 0.40 else
        'Low'       if prob >= 0.20 else 'Very Low'
    )
    return jsonify({
        'decision':        decision,
        'probability':     round(prob, 4),
        'probability_pct': f"{prob * 100:.1f}%",
        'risk_level':      risk_level,
        'threshold':       THRESHOLD,
        'explanations':    _generate_explanations(data, prob),
        'features_used':   list(FEATURE_DEFINITIONS.keys()),
        'demo_mode':       True,
    })


# ══════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html', features=FEATURE_DEFINITIONS)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    POST /api/predict
    Body: JSON  { "Attribute1": 2, "Attribute2": 18, ... }
    Returns:    { decision, probability, probability_pct, risk_level,
                  threshold, explanations, features_used, demo_mode? }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No JSON body received'}), 400

        # ── Fall back to demo if model not loaded ──────────────────────────
        if model is None or model_metadata is None:
            return _demo_predict(data)

        # ── Real model inference ───────────────────────────────────────────
        X        = preprocess_input(data)               # (1, n_features)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        model.eval()
        with torch.no_grad():
            logit = model(X_tensor)
            prob  = float(torch.sigmoid(logit).cpu().item())

        threshold  = model_metadata['threshold']
        decision   = 'DENIED' if prob >= threshold else 'APPROVED'
        risk_level = (
            'Very High' if prob >= 0.80 else
            'High'      if prob >= 0.60 else
            'Medium'    if prob >= 0.40 else
            'Low'       if prob >= 0.20 else 'Very Low'
        )

        return jsonify({
            'decision':        decision,
            'probability':     round(prob, 4),
            'probability_pct': f"{prob * 100:.1f}%",
            'risk_level':      risk_level,
            'threshold':       threshold,
            'explanations':    _generate_explanations(data, prob),
            'features_used':   list(model_metadata['selected_features']),
            'demo_mode':       False,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status':       'ok',
        'model_loaded': model is not None,
        'demo_mode':    model is None,
        'input_dim':    model_metadata['input_dim'] if model_metadata else None,
        'features':     model_metadata['selected_features'] if model_metadata else [],
    })


@app.route('/api/features', methods=['GET'])
def get_features():
    return jsonify(FEATURE_DEFINITIONS)


# ══════════════════════════════════════════════════════════════════════════
# Auto-load model at startup — works with both `python app.py` and gunicorn
# ══════════════════════════════════════════════════════════════════════════
def _auto_load():
    # 1. Explicit env override
    # 2. Default: models/best_credit_dnn.pt  (next to app.py)
    model_path = os.environ.get(
        'MODEL_PATH',
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'models', 'best_credit_dnn.pt')
    )
    if os.path.exists(model_path):
        try:
            load_model(model_path)
        except Exception as e:
            print(f"⚠️  Model load failed: {e}")
            print("   Running in DEMO mode.")
    else:
        print(f"ℹ️  Model not found at: {model_path}")
        print("   Copy best_credit_dnn.pt → models/best_credit_dnn.pt  then restart.")
        print("   Running in DEMO mode (heuristic predictions).")


_auto_load()   # called at import time so gunicorn workers load the model too

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)