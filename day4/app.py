from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Carregando o modelo e o scaler
model = joblib.load('modelo_churn.pkl')
scaler = joblib.load('scaler.pkl')

# Rota principal
@app.route('/')
def home():
	return render_template('index.html')

# Rota para predição
@app.route('/predict', methods=['POST'])
def predict():
	# Obtendo os dados do formulário
	int_features = [float(x) for x in request.form.values()]
	final_features = [np.array(int_features)]
    
	# Pré-processamento
	final_features = scaler.transform(final_features)
    
	# Realizando a previsão
	prediction = model.predict(final_features)
    
	output = prediction[0]
	return render_template('index.html', prediction_text='Previsão de Churn: {}'.format('Sim' if output == 1 else 'Não'))

if __name__ == "__main__":
	app.run(debug=True)
