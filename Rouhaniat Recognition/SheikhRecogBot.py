import cv2
from tensorflow.python.keras.models import load_model
import numpy as np
import telebot
from telebot import types
bot = telebot.TeleBot('Input Your Token')

# buttons = telebot.types.ReplyKeyboardMarkup(row_width=1)
# btn = telebot.types.KeyboardButton('SendPic')
# buttons.add(btn)

# markup = types.ReplyKeyboardMarkup(row_width=2)
# itembtn1 = types.KeyboardButton('a')
# markup.add(itembtn1)
# bot.send_message(chat_id, "Choose one letter:", reply_markup=markup)


@bot.message_handler(content_types=['photo'])
def photo(message):
    print('message.photo =', message.photo)
    fileID = message.photo[-1].file_id
    print('fileID =', fileID)
    file_info = bot.get_file(fileID)
    print('file.file_path =', file_info.file_path)
    downloaded_file = bot.download_file(file_info.file_path)

    with open(f"BotPhotos/{fileID}.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)

    model = load_model('sheykh.h5')

    image = cv2.imread(f"BotPhotos/{fileID}.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255
    image = image.reshape(1, 224, 224, 3)

    pred = model.predict([image])

    result = np.argmax(pred)

    if result == 1:
        print('از ما است')
        bot.reply_to(message, 'از ما است')

    elif result == 0:
        print('از ما نیست')
        bot.reply_to(message, 'از ما نیست')


@bot.message_handler(commands=['SendPic'])
def send_pic(message):
    bot.send_message(message.chat.id, 'Mrc', reply_markup=buttons)


@bot.message_handler(commands=['start'])
def say_hello(message):
    bot.send_message(message.chat.id, f'wellcome {message.from_user.first_name} Joonz')


@bot.message_handler(func=lambda message: True)
def send_unknown(message):
    bot.reply_to(message, 'نمیفهمم چی میگی یره!')


bot.polling()

