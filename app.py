import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "resume_hiring_model.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)


def build_model():
    rng = np.random.RandomState(42)
    n_samples = 500

    data = pd.DataFrame(
        {
            "experience_years": rng.randint(0, 12, n_samples),
            "skills_score": rng.randint(40, 100, n_samples),
            "interview_score": rng.randint(45, 100, n_samples),
            "communication_score": rng.randint(35, 100, n_samples),
            "education": rng.choice(["Bachelor", "Master", "PhD"], size=n_samples),
            "location": rng.choice(["Remote", "Hybrid", "Onsite"], size=n_samples),
        }
    )

    target = []
    for _, row in data.iterrows():
        score = (
            row["experience_years"] * 2.5
            + row["skills_score"] * 0.25
            + row["interview_score"] * 0.20
            + row["communication_score"] * 0.15
        )
        if row["education"] == "PhD":
            score += 12
        elif row["education"] == "Master":
            score += 6
        if row["location"] == "Remote":
            score += 4
        elif row["location"] == "Hybrid":
            score += 2
        target.append(1 if score + rng.uniform(-10, 10) > 155 else 0)

    data["hired"] = target

    numeric_features = ["experience_years", "skills_score", "interview_score", "communication_score"]
    categorical_features = ["education", "location"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, max_depth=6)),
        ]
    )

    model.fit(data.drop(columns=["hired"]), data["hired"])
    joblib.dump(model, MODEL_PATH)
    return model


def load_model():
    if not os.path.exists(MODEL_PATH):
        return build_model()
    return joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    if request.method == "POST":
        model = load_model()
        form_data = {
            "experience_years": int(request.form.get("experience_years", 0)),
            "skills_score": int(request.form.get("skills_score", 0)),
            "interview_score": int(request.form.get("interview_score", 0)),
            "communication_score": int(request.form.get("communication_score", 0)),
            "education": request.form.get("education", "Bachelor"),
            "location": request.form.get("location", "Remote"),
        }
        input_df = pd.DataFrame([form_data])
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        prediction = "Hired" if pred == 1 else "Not Hired"
        probability = round(float(prob) * 100, 1)

    return render_template("index.html", prediction=prediction, probability=probability)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
