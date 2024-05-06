import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
import boto3
from botocore.exceptions import NoCredentialsError
import requests
from pathlib import Path
import pymongo
import json


class Bot:

    def __init__(self, token, telegram_chat_url):
        self.telegram_bot_client = telebot.TeleBot(token)
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60)

        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    @staticmethod
    def is_current_msg_photo(msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path)
        )

    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')

class QuoteBot(Bot):
    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if msg["text"] != 'Please don\'t quote me':
            self.send_text_with_quote(msg['chat']['id'], msg["text"], quoted_msg_id=msg["message_id"])


class ObjectDetectionBot(Bot):

    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if self.is_current_msg_photo(msg):
            # Download the user photo
            photo_path = self.download_user_photo(msg)
            print('Photo successfully downloaded')

            # Upload the photo to S3
            try:
                s3_client = boto3.client('s3')
                s3_key = 'image.jpeg'
                s3_bucket = os.environ['BUCKET_NAME']
                print('Uploading...')
                s3_client.upload_file(photo_path, s3_bucket, s3_key)
                print('File successfully uploaded')

            except NoCredentialsError:
                print('Credentials not available')

            # Send a request to the `yolo5` service for prediction
            url = 'http://yolo5:8081/predict'
            params = {'imgName': 'image.jpeg'}
            response = requests.post(url, params=params)

            # Check if the request was successful (status code 200)
            # Download predicted image from S3 and send it to the user
            print(response.content)
            predicted_img_path = 'predicted_image.jpeg'
            s3_client.download_file(s3_bucket, f'predicted/{s3_key}', predicted_img_path)
            self.send_photo(msg['chat']['id'], predicted_img_path)
            # send summery text
            response = requests.post(url, params=params)
            formatted_response = self.formatted_message(response.text)
            print('Response:', formatted_response)
            self.send_text(msg['chat']['id'], text=formatted_response)
            print('Prediction successful!')


    def formatted_message(self, json_ob):
        obj_count = {}
        formatted_string = f"Detected Objects:\n"
        for label in json_ob["labels"]:
            class_name = label["class"]
            if class_name in obj_count:
                obj_count[class_name] += 1
            else:
                obj_count[class_name] = 1
        for key, value in obj_count.items():
            formatted_string += f"{key}: {value}\n"
        return formatted_string
