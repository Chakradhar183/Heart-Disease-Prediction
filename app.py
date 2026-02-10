from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# -------------------------
# Load model, scaler, and encoders
# -------------------------
model = joblib.load("heartdisease_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder_data = joblib.load("encoders.pkl")

# Extract feature names and label mappings
FEATURES = encoder_data['feature_names']
label_mappings = encoder_data['label_mappings']

print("="*70)
print("HEART DISEASE PREDICTION WEB APP")
print("="*70)
print("Model loaded successfully")
print(f"Model expects {len(FEATURES)} features:")
for i, feat in enumerate(FEATURES, 1):
    print(f"  {i}. {feat}")
print("="*70)

# -------------------------
# Helper function to build display dictionary
# -------------------------
def build_display_dict(raw_inputs):
    """Convert numeric encoded values back to human-readable labels"""
    display = {}
    encoders = encoder_data['encoders']
    
    for feat in FEATURES:
        val = raw_inputs[feat]
        # Try to decode categorical features
        if feat in encoders and feat != 'HeartDisease':
            try:
                # Convert back to original label
                encoder = encoders[feat]
                if hasattr(encoder, 'inverse_transform'):
                    decoded = encoder.inverse_transform([int(float(val))])[0]
                    display[feat] = str(decoded)
                else:
                    display[feat] = str(val)
            except:
                display[feat] = str(val)
        else:
            # Numeric features - format nicely
            try:
                fval = float(val)
                if fval.is_integer():
                    display[feat] = str(int(fval))
                else:
                    display[feat] = f"{fval:.2f}"
            except:
                display[feat] = str(val)
    
    return display

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return render_template("index.html", 
                         features=FEATURES,
                         label_mappings=label_mappings)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect all feature values from form
        raw_inputs = {}
        for feat in FEATURES:
            val = request.form.get(feat)
            if val is None or val == "":
                raise ValueError(f"Missing value for feature: {feat}")
            raw_inputs[feat] = val
        
        # Convert to numeric array in correct order
        numeric_values = []
        for feat in FEATURES:
            try:
                numeric_values.append(float(raw_inputs[feat]))
            except ValueError:
                raise ValueError(f"Invalid numeric value for {feat}: '{raw_inputs[feat]}'")
        
        # Prepare input array
        X = np.array(numeric_values).reshape(1, -1)
        
        # Validate dimensions
        if X.shape[1] != len(FEATURES):
            raise ValueError(
                f"Input has {X.shape[1]} features, but model expects {len(FEATURES)} features."
            )
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        prediction = int(model.predict(X_scaled)[0])
        
        # Get confidence probability
        confidence = "N/A"
        try:
            proba = model.predict_proba(X_scaled)[0]
            confidence = round(float(proba[prediction]) * 100, 2)
        except:
            pass
        
        # Build display dictionary
        display = build_display_dict(raw_inputs)
        
        # Determine result text
        result_text = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        
        return render_template("result.html", 
                             prediction=result_text,
                             confidence=confidence,
                             features=display)
    
    except Exception as e:
        return render_template("index.html", 
                             error=str(e),
                             features=FEATURES,
                             label_mappings=label_mappings)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for programmatic predictions"""
    try:
        data = request.get_json(force=True)
        
        # Validate all features present
        numeric_values = []
        for feat in FEATURES:
            if feat not in data:
                raise ValueError(f"Missing feature: {feat}")
            numeric_values.append(float(data[feat]))
        
        X = np.array(numeric_values).reshape(1, -1)
        
        if X.shape[1] != len(FEATURES):
            raise ValueError(f"Expected {len(FEATURES)} features, got {X.shape[1]}")
        
        X_scaled = scaler.transform(X)
        prediction = int(model.predict(X_scaled)[0])
        
        # Get probability
        probability = None
        try:
            proba = model.predict_proba(X_scaled)[0]
            probability = float(proba[1])
        except:
            pass
        
        result_text = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        
        return jsonify({
            "prediction": prediction,
            "probability": probability,
            "result": result_text
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -------------------------
# Run the App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
