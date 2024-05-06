import time
from pathlib import Path
from flask import Flask, request
from detect import run
import uuid
import yaml
from loguru import logger
import os
import boto3
from botocore.exceptions import NoCredentialsError
from pymongo import MongoClient

# Initialize MongoDB client
mongo_client = MongoClient('mongodb://mongo1:27017/')
db = mongo_client['predictions']
collection = db['predicti1on_summaries']
#ENV
images_bucket = os.environ['BUCKET_NAME']

# Load class names from coco128.yaml
with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Generates a UUID for this current prediction HTTP request.
    prediction_id = str(uuid.uuid4())
    logger.info(f'Prediction ID: {prediction_id}. Processing started.')

    # Receive the image name from the request URL parameter
    img_name = request.args.get('imgName')

    # Download the image from S3
    s3 = boto3.client('s3')
    original_img_path = img_name
    try:
        s3.download_file(images_bucket, img_name, original_img_path)
        logger.info(f'Prediction ID: {prediction_id}/{img_name}. Image downloaded successfully.')
    except NoCredentialsError:
        logger.error('AWS credentials not available. Image download failed.')
        return 'AWS credentials not available', 500

    # Perform object detection on the image
    run(
        weights='yolov5s.pt',
        data='data/coco128.yaml',
        source=original_img_path,
        project='usr/src/app',
        name=prediction_id,
        save_txt=True
    )
    logger.info(f'Prediction ID: {prediction_id}/{original_img_path}. Object detection completed.')

    # Upload the predicted image to S3
    predicted_img_path = Path(f'usr/src/app/{prediction_id}/{original_img_path}')
    try:
        s3.upload_file(str(predicted_img_path), images_bucket, f'predicted/{img_name}')
        logger.info(f'Prediction ID: {prediction_id}/{img_name}. Predicted image uploaded to S3 successfully.')
    except NoCredentialsError:
        logger.error('AWS credentials not available. Image upload failed.')
        return 'AWS credentials not available', 500

    # Parse prediction labels and create a summary
    pred_summary_path = Path(f'usr/src/app/{prediction_id}/labels/{original_img_path.split(".")[0]}.txt')
    if pred_summary_path.exists():
        with open(pred_summary_path) as f:
            labels = f.read().splitlines()
            labels = [line.split(' ') for line in labels]
            labels = [{
                'class': names[int(l[0])],
                'cx': float(l[1]),
                'cy': float(l[2]),
                'width': float(l[3]),
                'height': float(l[4]),
            } for l in labels]

        logger.info(f'Prediction ID: {prediction_id}/{original_img_path}. Prediction summary:\n{labels}')
        predicted_img_path_str = str(predicted_img_path)
        prediction_summary = {
            'prediction_id': prediction_id,
            'original_img_path': original_img_path,
            'predicted_img_path': predicted_img_path_str,
            'labels': labels,
            'time': time.time()
        }

        # Store the prediction summary in MongoDB
        collection.insert_one(prediction_summary)
        logger.info(f'Prediction ID: {prediction_id}. Prediction summary stored in MongoDB.')

        return prediction_summary, 200
    else:
        logger.error(f'Prediction ID: {prediction_id}/{original_img_path}. Prediction result not found.')
        return f'Prediction ID: {prediction_id}/{original_img_path}. Prediction result not found', 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081, debug=True)