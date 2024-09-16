from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import qrcode
from io import BytesIO
from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode

app = Flask(__name__)

def create_qr_code(data, size):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=1,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img_qr = qr.make_image(fill='black', back_color='white').convert('L')
    img_qr = img_qr.resize((size, size))
    qr_array = np.array(img_qr, dtype=np.uint8)
    return qr_array

def embed_qr_in_logo(img, qr_img):
    h, w, _ = img.shape
    qr_h, qr_w = qr_img.shape
    if qr_h > h or qr_w > w:
        raise ValueError("QR code is too large to fit in the image")
    for i in range(qr_h):
        for j in range(qr_w):
            pixel = img[i, j, 0]
            qr_bit = qr_img[i, j] // 255
            img[i, j, 0] = (pixel & 0xFE) | qr_bit
    return img

def extract_qr_from_logo(img, size):

    qr_img = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            pixel = img[i, j, 0]
            qr_bit = pixel & 0x1
            qr_img[i, j] = 255 * qr_bit
    
    return qr_img

def decode_qr_code(image):
    # Ensure image is in RGB format
    image = Image.fromarray(image)
    image = image.convert('RGB')  # Convert to RGB if not already
    
    # Decode the QR code
    decoded_objects = pyzbar_decode(image)
    
    decoded_message = None
    for obj in decoded_objects:
        decoded_message = obj.data.decode('utf-8')
        break  # Assuming there's only one QR code, exit after finding the first one
    
    return decoded_message

@app.route('/encode', methods=['POST'])
def encode():
    # Retrieve the message to be embedded in the QR code
    data = request.form['message']
    
    # Read the uploaded image file (as in-memory binary)
    img_file = request.files['image'].read()
    
    # Convert the image from binary to a NumPy array and then decode it as BGR using OpenCV
    img_array = np.frombuffer(img_file, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Create the QR code and resize it to fit within the image
    qr_img = create_qr_code(data, min(img.shape[:2]))

    # Embed the QR code using LSB steganography in the image
    img_with_qr = embed_qr_in_logo(img, qr_img)

    # Save the image with the embedded QR code to an in-memory buffer
    _, buffer = cv2.imencode('.png', img_with_qr)
    io_buf = BytesIO(buffer)

    # Send the image as a file response in PNG format
    return send_file(io_buf, mimetype='image/png', as_attachment=True, download_name='encoded_image.png')


@app.route('/decode', methods=['POST'])
def decode():
    # Read the uploaded image file
    img_file = request.files['image'].read()
    
    # Use OpenCV to decode the in-memory image (BGR format)
    img_array = np.frombuffer(img_file, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Extract the hidden QR code from the image
    qr_img = extract_qr_from_logo(img, min(img.shape[:2]))

    # Decode the extracted QR code (you may need to adjust for BGR-to-RGB if required)
    decoded_message = decode_qr_code(qr_img)

    return jsonify({'message': decoded_message})


if __name__ == '__main__':
    app.run(debug=True)
