import numpy as np
import flask as Flask,requests,jsonify,render_template
import pickle


app=Flask(__name__) 
model=pickle.load(open("model.pkl","rb"))
@app.route('/')
def home():
	return render_template("home.html")
@app.route('/predict',method=["POST"])
def predict():
	int_feature=[int(x) for x in request.form.values()]
	final_feature=[np.array('int_feature')]
	prediction=model.predict (final_feature)
	output=round(prediction[0],2)
	return render_template('index.html',prediction_text='Car condition is{}'.format(output))
if __name__ == '__main__':
    app.run(debug=True)
