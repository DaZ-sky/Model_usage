import io
import sys
import base64
import numpy as np
import torchvision
from PIL import Image
import tritonclient.http as triton_http
import torchvision.transforms as transforms

def pic_to_str(pic_path):
    with open(pic_path, "rb") as image:
        b64string = base64.b64encode(image.read()).decode('ASCII')
    return b64string

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def prepare_pic(pic_str, image_size =(32, 32)):
    f = io.BytesIO(base64.b64decode(pic_str))
    img = Image.open(f)

    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    transformed_image = image_transform(img).unsqueeze(dim=0)
    return transformed_image, to_numpy(transformed_image)

def get_class(
    triton_client: triton_http.InferenceServerClient,
    model_name: str,
    model_version: str,
    image_path: str
) -> np.ndarray:

    input_img = np.array(prepare_pic(pic_to_str(image_path)))
    input_img = np.expand_dims(input_img.squeeze()[0], axis=0)
    print(input_img.shape)
    input_tensor = triton_http.InferInput(
        name="input",
        shape=list(input_img.shape),
        datatype=triton_http.np_to_triton_dtype(input_img.dtype),
    )
    input_tensor.set_data_from_numpy(input_img)

    response = triton_client.infer(
        model_name=model_name,
        model_version=model_version,
        inputs=[input_tensor],
    )
    return response.as_numpy(name="output")


def main() -> None:
    triton_client = triton_http.InferenceServerClient(
        url="127.0.0.1:8000"
    )

    args = sys.argv
    
    model_name = args[1]
    model_version = args[2]
    image_path = args[3]

    print("Send test request")
    print("Wait for response...")

    if (
        class_ := get_class(triton_client, model_name, model_version, image_path)
    ) is None:
        print("Model is not ready... Check your server")
        return
    
    print("Got class from server")
    print(class_)
    return class_

    
if __name__ == "__main__":
    main()