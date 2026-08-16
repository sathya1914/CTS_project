from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import pandera as pa
from pandera.errors import SchemaError
import io

app = FastAPI(title="Data Quality & Anomaly Detection API")

# Enable CORS for HTML/CSS Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/process-data")
async def process_data(
    file: UploadFile = File(...),
    contamination: float = Form(0.05),
    require_numeric: bool = Form(True)
):
    try:
        # 1. Read Uploaded CSV File
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")
        
        total_rows = len(df)
        
        # 2. Schema / Validation Rules (Pandera)
        # Dynamic Schema checks null values & numeric types on primary columns
        validation_flags = [True] * total_rows
        schema_errors = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            raise HTTPException(status_code=400, detail="CSV must contain at least one numeric column for outlier detection.")

        # Flag rows with missing values
        null_mask = df.isnull().any(axis=1)
        for idx in df[null_mask].index:
            validation_flags[idx] = False
            schema_errors.append({"row": int(idx), "reason": "Missing values detected"})

        # Clean numerical subset for ML model
        clean_df_for_ml = df[numeric_cols].fillna(df[numeric_cols].median())

        # 3. Machine Learning Outlier Detection (Isolation Forest)
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        # Fit-predict (-1 indicates outlier, 1 indicates inlier)
        predictions = iso_forest.fit_predict(clean_df_for_ml)
        anomaly_scores = iso_forest.decision_function(clean_df_for_ml)

        # 4. Consolidate Pipeline Results
        processed_rows = []
        valid_count = 0
        outlier_count = 0
        validation_failed_count = 0

        # Choose top two numeric columns for 2D visual scatter plot
        x_col = numeric_cols[0]
        y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]

        for i in range(total_rows):
            is_valid_schema = validation_flags[i]
            is_outlier = predictions[i] == -1
            
            if not is_valid_schema:
                status = "Validation Failed"
                validation_failed_count += 1
            elif is_outlier:
                status = "Outlier Flagged"
                outlier_count += 1
            else:
                status = "Clean"
                valid_count += 1

            processed_rows.append({
                "id": i,
                "x": float(df[x_col].iloc[i]) if pd.notnull(df[x_col].iloc[i]) else 0,
                "y": float(df[y_col].iloc[i]) if pd.notnull(df[y_col].iloc[i]) else 0,
                "status": status,
                "anomaly_score": round(float(anomaly_scores[i]), 4),
                "raw_data": df.iloc[i].to_dict()
            })

        pass_rate = round((valid_count / total_rows) * 100, 1)

        return {
            "summary": {
                "total_rows": total_rows,
                "valid_count": valid_count,
                "outlier_count": outlier_count,
                "validation_failed_count": validation_failed_count,
                "pass_rate": pass_rate,
                "x_axis_col": x_col,
                "y_axis_col": y_col
            },
            "data": processed_rows
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_index():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)