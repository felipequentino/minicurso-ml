from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Carregando o modelo e o scaler
model = joblib.load('../best_model_random_forest.pkl')
scaler = joblib.load('../scaler.pkl')

# Lista completa de colunas que o modelo espera (excluindo 'customerID' e 'Churn')
model_columns = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Female', 'gender_Male',
    'Partner_No', 'Partner_Yes',
    'Dependents_No', 'Dependents_Yes',
    'PhoneService_No', 'PhoneService_Yes',
    'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_No', 'PaperlessBilling_Yes',
    'PaymentMethod_Bank transfer (automatic)',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

def set_dummy(input_data, column_value):
    if column_value in model_columns:
        input_data[column_value] = 1

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recebe dados JSON
        data = request.get_json()

        # Inicializa um dicionário com todas as colunas como 0
        input_data = {col: 0 for col in model_columns}

        # Extrai e valida os valores numéricos
        try:
            tenure = float(data['tenure'])
            monthly_charges = float(data['MonthlyCharges'])
            total_charges = float(data['TotalCharges'])
        except ValueError:
            return jsonify({'error': 'Valores numéricos inválidos'}), 400

        # Prepara os dados numéricos para escalonamento
        numerical_features = np.array([[tenure, monthly_charges, total_charges]])
        numerical_features_scaled = scaler.transform(numerical_features)

        # Atualiza as características numéricas escalonadas no dicionário
        input_data['tenure'] = numerical_features_scaled[0][0]
        input_data['MonthlyCharges'] = numerical_features_scaled[0][1]
        input_data['TotalCharges'] = numerical_features_scaled[0][2]

        # Define as dummies com base nas seleções
        for key, value in data.items():
            set_dummy(input_data, value)

        # Converter o dicionário em um array na ordem correta
        final_features = [input_data[col] for col in model_columns]
        final_features = np.array([final_features])

        # Realizando a previsão
        prediction = model.predict(final_features)

        return jsonify({
            'prediction': int(prediction[0])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
