import os
from io import BytesIO
from flask import Flask, request, send_from_directory, send_file, jsonify
from flask_pymongo import PyMongo
from PIL import Image
import numpy as np
import glob
import json
import subprocess
import skimage

app = Flask(__name__, static_url_path='/static', static_folder = "./frontend/public")
# This is the app config I used
# app.config['MONGO_URI'] = "mongodb://host.docker.internal:27017/main"
app.config['MONGO_URI'] = "mongodb://%s:27017/main"%(os.environ['DB_PORT_27017_TCP_ADDR'])

mongo = PyMongo(app)
db = mongo.db

def display_array(arr):
    """
    Display an image represented as an array
    """
    img = Image.fromarray(arr.astype(np.uint8))
    bytesIO = BytesIO()
    img.save(bytesIO, 'PNG')
    bytesIO.seek(0)
    return bytesIO

def convert_to_array(path):
    """
    Converts an image file to an array for later processing
    """
    image_file = Image.open(path).convert("RGB")
    image_array = np.array(image_file)
    return image_array

def apply_grayscale_filter(img_data):
    """
    Converts a colored image to grascale
    """
    for i in range(len(img_data)):
        for j in range(len(img_data[i])):
            blue = img_data[i, j, 0]
            green = img_data[i, j, 1]
            red = img_data[i, j, 2]

            greyscale = blue * 0.114 + green * 0.587 + red * 0.299
            img_data[i, j] = greyscale

    return img_data

def crop_image(img_data):
    """
    Crops the image to the maximum square crop from the centre of the image
    """
    img_data = apply_grayscale_filter(img_data)
    y, x, z = img_data.shape
    if x == y:
        return img_data
    if x > y:
        x_crop = x - y
        cropped_data = img_data[: , x_crop // 2: x - x_crop // 2]
        return cropped_data
    elif y > x:
        y_crop = y - x
        cropped_data = img_data[y_crop // 2: y - y_crop // 2, :]
        return cropped_data

def crop_for_normalizing(np_data):
    """
    Crops the image dimensions to the highest multiple of 32 for creating 32 x 32 patches
    """
    y, x, z = np_data.shape

    if y % 2 != 0:
        y -= 1
    if x % 2 != 0:
        x -= 1
    x_crop = x % 32
    y_crop = y % 32

    cropped_data = np_data[y_crop // 2: y - y_crop // 2, x_crop // 2: x - x_crop // 2]

    return cropped_data

def downsample_image(img_data, x, filtertype):
    """
    Converts the image data to grayscale first, then resizes the image down by the donwsample factor x provided
    """
    if filtertype != "original":
        img_data = apply_grayscale_filter(img_data)
    img = Image.fromarray(img_data)
    new_width = int(img_data.shape[1] // x)
    new_height = int(img_data.shape[0] // x)
    new_dimensions = (new_width, new_height)

    # Downsizing the image using the nearest neighbour resampling method
    downsampled_img = img.resize(new_dimensions, resample=Image.NEAREST)

    return np.asarray(downsampled_img)

def normalize_data(img_data):
    """
    This method performs the following operation in order to transform the image data:

    1.  Splits up the image into multiple 32x32 patches
    2.  For each patch, computes the mean and standard deviation of image values
    3.  Computes u = median of the means and v = median of the standard deviations** of all patches
    4.  Rescales the image values so that the minimum value of u - 3*v and the maximum value of u + 3*v map to 0 and 255
        respectively
    """
    crop_data = crop_for_normalizing(img_data)
    img_data = apply_grayscale_filter(crop_data)

    patches = skimage.util.view_as_blocks(img_data, (32, 32, 3))
    avg = []
    std = []
    
    for patch in patches:
        patch_matrix = patch[0][0]
        avg.append(np.mean(patch_matrix))
        std.append(np.std(patch_matrix))
        
    u = np.median(avg)
    v = np.median(std)
    upper_limit = u + 3 * v
    lower_limit = u - 3 * v

    def mapping_func(i):
        if i < lower_limit:
            return 0
        if i > upper_limit:
            return 255
        else:
            return i
        
    contrast_normalize = np.vectorize(mapping_func)
    normalized_data = contrast_normalize(img_data)

    return normalized_data.astype(np.uint8)

def power_spectrum(img_data):
    """
    Takes the largest square crop from the centre of the image. Then, computes the Fourier transform of that cropped
    image, takes the square of the absolute value of each (complex valued) Fourier component, normalizes to the range
    0-255, and returns the result.
    """
    img_data = crop_image(img_data)
    fourier_transform = np.fft.fft(img_data)
    fourier_transform_squared = np.square(np.abs(fourier_transform))

    return normalize_data(fourier_transform_squared)

@app.route("/", defaults={'path': ''})
@app.route("/<path:path>")
def base(path):
  return send_from_directory('frontend/public', 'index.html')

@app.route("/api/populate_database", methods=['GET'])
def populate_database():
    """
    Finds all images files of type .png inside the assets folder and for each image, inserts into the database images
    collection a document with the following fields:
    file_name, width, height, area, size_bytes
    Creates an index on the 'width' field for better search and sorting performance.
    """
    if request.method == "GET":
        file_path = "./assets/"
        img_files = glob.glob(file_path + "*.png")
        img_documents = []

        for img_file in img_files:
            img_size = os.stat(img_file).st_size
            img = Image.open(img_file)
            width, height = img.size
            img_document = {
                "file_name": str(img_file.replace(file_path, "")),
                "width": width,
                "height": height,
                "area": width * height,
                "size_bytes": img_size
            }
            img_documents.append(img_document)

        db.images.insert_many(img_documents)
        # Creating an index on the "width" filed
        db.images.create_index("width")
        response = "database populated successfully"

        return jsonify(response), 201

@app.route("/api/all", methods=['GET'])
def image_content():
    """
    Serves all the contents of the images collection in JSON format.
    """
    if request.method == "GET":
        all_images = db.images.find({}, {'_id': False})
        img_info_list = [img for img in all_images]

        return json.dumps(img_info_list)

@app.route("/api/query", methods=['GET'])
def get_images():
    """
    Serves a list of all available image file names, that match query parameters of maximum width, minimum area and
    maximum image size in bytes.
    """
    if request.method == "GET":
        parameters = request.args

        max_width = int(parameters['max_width'])
        min_area = int(parameters['min_area'])
        max_size_bytes = int(parameters['max_size_bytes'])

        images = db.images.find({"width": {"$lte": max_width}, "area": {"$gte": min_area}, "size_bytes": {"$lte": max_size_bytes}},
                                {"file_name": True, "_id": False})
        img_list = [img["file_name"] for img in images]

        return json.dumps(img_list)

@app.route("/api/filter/<imagename>/<filtertype>", methods=['GET'])
def transform_image(imagename, filtertype):
    """
    Transforms the image based on the following filters:

    1. `original`: returns the original image as a png
    2. `grayscale`: returns the grayscale version of the image as a png
    3. `crop`: returns the largest possible square crop from the center of the image
    4. `downsample`: returns a `value` downsampled version of the image (i.e. if value=1.5, the output is 1.5x smaller
        than the input)
    5. `normalize`: performs robust contrast normalization using patches
    """
    if request.method == "GET":
        # x is the factor by which the image will be scaled
        x = float(request.args.get("value", default=None, type=None))

        # Check if full image name is provided, else and the '.png' extension
        if imagename[-4:] != ".png":
            imagename += ".png"
        file_path = "./assets/" + imagename
        img_data = convert_to_array(file_path)

        if filtertype == "original":
            img_data = img_data
        if filtertype == "grayscale":
            img_data = apply_grayscale_filter(img_data)
        if filtertype == "crop":
            img_data = crop_image(img_data)
        if filtertype == "normalize":
            img_data = normalize_data(img_data)
        if filtertype == "power_spectrum":
            img_data = power_spectrum(img_data)
        if x and x != 0:
            img_data = downsample_image(img_data, x, filtertype)

        return send_file(display_array(img_data), mimetype="png")

@app.route("/api/backup", methods=['GET'])
def generate_backup():
    """
    Uses the subprocess module to call zip to create a zip of all the images in the 'assets' directory and store it in
    the 'backups' directory
    """
    if request.method == "GET":
        dir = os.path.dirname(os.path.abspath(__file__))
        # The instructions refer to the 'images' directory, but the assignment directory name is set to 'assets'
        file_path1 = "assets"
        file_path2 = os.path.join(dir, "backups/")
        zip_file_name = "assets"

        command = [f"zip -r {zip_file_name}.zip {file_path1}; mv {zip_file_name}.zip {file_path2}"]
        process = subprocess.run(command, shell=True, capture_output=True)

        if process.returncode == 0:
            return jsonify("back up complete"), 200
        else:
            return jsonify("back up failed"), 417

if __name__ == "__main__":
  app.run(host='0.0.0.0', debug=True)