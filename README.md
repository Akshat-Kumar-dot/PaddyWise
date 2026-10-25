# PaddyWise — Paddy Yield Prediction & Variety Recommendation

ML Project · B1_Sec_A_Sem_V · Group 2 (Akshat Kumar, Kushal Bajaj, Chirag Sahu)

| File | What it is |
|---|---|
| `Paddy_Yield_Prediction.ipynb` | Full ML workflow: cleaning → EDA → feature optimization → 5 models → test metrics → key factors → recommendation |
| `Hybrid_Model_From_Scratch.ipynb` | The hybrid stacking model hand-coded in NumPy, explained step by step and checked against scikit-learn |
| `paddy_model.pkl` | Trained hybrid stacking model (created by the notebook) |
| `app.py` | Streamlit web app (front end) |
| `paddydataset.csv` | UCI Paddy Dataset (https://doi.org/10.24432/C55W3J) |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501

## Deploy free on Streamlit Community Cloud
1. Push this folder to a new public GitHub repository (all files, including `paddy_model.pkl` and `.streamlit/`).
2. Go to https://share.streamlit.io → sign in with GitHub → **Create app**.
3. Choose the repository, branch `main`, main file `app.py` → **Deploy**.
4. You get a public link like `https://<your-app>.streamlit.app` to share in the evaluation.

If you re-run the notebook, it overwrites `paddy_model.pkl`. Push the new file too.
