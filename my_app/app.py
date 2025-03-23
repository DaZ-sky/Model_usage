import io
import base64
import torchvision
from PIL import Image
from flask import Flask, request, jsonify
import torchvision.transforms as transforms

app = Flask(__name__)

@app.route('/pic', methods=['GET', 'POST'])
def preprocess_image():
    data = request.get_json()
    image_size =(32, 32)
    pic_str = data.get('pic')
    f = io.BytesIO(base64.b64decode(pic_str))
    img = Image.open(f)

    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    transformed_image = image_transform(img).unsqueeze(dim=0)
    return jsonify({"pic": to_numpy(transformed_image).tolist()}), 200

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


@app.route('/test', methods=['GET'])
def test():
    return "TEST"

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5000)