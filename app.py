from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from model import CBmodel, predict, phoBERT, label, LABEL_SIZE
import torch

model = CBmodel(phoBERT, 768, LABEL_SIZE)
model = torch.load("D:\Flask fundamental\model\CBmodel_Ver1.pth", map_location=torch.device('cpu'))
model.eval()
app = Flask(__name__)
CORS(app)
@app.route('/')
@cross_origin("*")
def index():
    return render_template('index.html')

@app.route('/api/chatbox', methods=['POST'])
def chatbox():
    message = request.json['message']
    output = predict(message, model)
    
   
    return jsonify({'message': output})
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)