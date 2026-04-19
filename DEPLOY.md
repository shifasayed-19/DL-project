# CreditIQ — Heroku Deployment Guide

## Project Structure

```
credit_scoring_app/
├── app.py                  ← Flask app (gunicorn-compatible)
├── Procfile                ← Heroku process declaration
├── runtime.txt             ← Python 3.11.9
├── requirements.txt        ← CPU-only torch + all deps
├── .gitignore
├── models/
│   └── best_credit_dnn.pt  ← ⬅ Copy your trained model here before deploy
└── templates/
    └── index.html
```

---

## Step-by-Step Heroku Deployment

### Prerequisites
```bash
# Install Heroku CLI
# macOS:
brew tap heroku/brew && brew install heroku

# Windows: download from https://devcenter.heroku.com/articles/heroku-cli
# Ubuntu:
curl https://cli-assets.heroku.com/install.sh | sh

# Verify
heroku --version
```

---

### 1. Copy your trained model into the project
```bash
# From your notebook output folder:
cp ./credit_scoring_project/models/best_credit_dnn.pt ./models/best_credit_dnn.pt
```

---

### 2. Initialise a Git repo
```bash
cd credit_scoring_app
git init
git add .
git commit -m "Initial commit — CreditIQ credit scoring app"
```

---

### 3. Login to Heroku and create the app
```bash
heroku login
heroku create creditiq-scoring
# Replace 'creditiq-scoring' with your preferred app name
```

---

### 4. Deploy

```bash
git push heroku main
```

If your local branch is `master` instead of `main`:
```bash
git push heroku master
```

---

### 5. Scale and open
```bash
heroku ps:scale web=1
heroku open
```

Your app will be live at:
`https://creditiq-scoring.herokuapp.com`

---

### 6. Check logs if anything goes wrong
```bash
heroku logs --tail
```

---

## Important Notes for the Model File

Heroku's **slug size limit is 500 MB**. PyTorch model files can be large.

### Option A — Include model in Git (simplest, works if model < ~100 MB)
Just copy `best_credit_dnn.pt` into `models/` and commit it. Remove `models/*.pt` from `.gitignore` first:

```bash
# In .gitignore, comment out or delete this line:
# models/*.pt

git add models/best_credit_dnn.pt
git commit -m "Add trained model"
git push heroku main
```

### Option B — Heroku Large File Storage (model > 100 MB)
Use AWS S3 or Cloudflare R2 to host the model, then download it on startup.

Add to the top of `app.py`:
```python
import urllib.request

def _download_model_if_needed():
    path = os.path.join(os.path.dirname(__file__), 'models', 'best_credit_dnn.pt')
    if not os.path.exists(path):
        url = os.environ.get('MODEL_URL')   # set this in Heroku config vars
        if url:
            print("Downloading model from remote...")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(url, path)
            print("Model downloaded.")
```

Then set the env variable on Heroku:
```bash
heroku config:set MODEL_URL=https://your-bucket.s3.amazonaws.com/best_credit_dnn.pt
```

---

## Local Testing Before Deploy

```bash
# Install deps
pip install -r requirements.txt

# Run with gunicorn (same as Heroku)
gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:5000

# Or with Flask dev server
python app.py
```

Open: http://localhost:5000

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | Set automatically by Heroku |
| `MODEL_PATH` | `./models/best_credit_dnn.pt` | Override model file location |
| `MODEL_URL` | — | Optional: URL to download model from |

Set on Heroku:
```bash
heroku config:set MODEL_PATH=/app/models/best_credit_dnn.pt
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: torch` | Check `requirements.txt` has the `--extra-index-url` line |
| `H10 App crashed` | Run `heroku logs --tail` to see the error |
| `R14 Memory exceeded` | Upgrade to Heroku Standard-2X dyno (`heroku ps:type web=standard-2x`) |
| Model loads but predicts wrong | Check `MODEL_PATH` is correct and file isn't corrupted |
| Slug too large | Use Option B (remote model download) |

---

## Heroku Dyno Requirements

The CPU-only PyTorch build is ~500 MB installed. Recommended:

- **Eco / Basic dyno** — fine for demo/testing
- **Standard-1X** — minimum for production (512 MB RAM)
- **Standard-2X** — recommended (1 GB RAM), avoids R14 memory errors

```bash
# Upgrade dyno type
heroku ps:type web=standard-1x
```
