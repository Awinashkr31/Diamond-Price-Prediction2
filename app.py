from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# 1. Load the pre-trained model pipeline if available
model_path = 'model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # 2. Retrieve form input values
            carat = float(request.form.get('carat'))
            cut = request.form.get('cut')
            color = request.form.get('color')
            clarity = request.form.get('clarity')
            depth = float(request.form.get('depth'))
            table = float(request.form.get('table'))
            x_val = float(request.form.get('x'))
            y_val = float(request.form.get('y'))
            z_val = float(request.form.get('z'))
            
            # 3. Create a DataFrame for prediction
            input_data = pd.DataFrame({
                'carat': [carat],
                'cut': [cut],
                'color': [color],
                'clarity': [clarity],
                'depth': [depth],
                'table': [table],
                'x': [x_val],
                'y': [y_val],
                'z': [z_val]
            })
            
            # 4. Make a prediction
            if model:
                pred = model.predict(input_data)
                prediction = f"Predicted Price: ${round(pred[0], 2)}"
            else:
                prediction = "Model not found. Please train and save your model as model.pkl."
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
