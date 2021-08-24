import os.path
import random
import time

import js2py
import requests

generate_captcha_key = js2py.eval_js(
    "function generate_captcha_key() {	return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {		var r = Math.random()*16|0, v = c == 'x' ? r : (r&0x3|0x8);		return v.toString(16);	}).replace(/-/g,'');}")

proxies = {
    "http": '172.16.10.111:18118',
    "https": '172.16.10.111:18118'
}


def download_captcha_image(output_path):
    url = f'https://gop.captcha.garena.com/image?key={generate_captcha_key()}'
    print(url)
    res = requests.get(url, proxies=proxies)
    content = res.content

    if len(content) < 500:
        return False

    with open(output_path, 'wb') as f:
        f.write(content)
    return True


output_dir = '/media/data_it/Data_set/database_image/ocr/captcha/garena'
n = 5000
for i in range(n):
    filepath = os.path.join(output_dir, f'{i:0>5}.jpg')
    if os.path.exists(filepath):
        continue
    res = download_captcha_image(filepath)
    print(f'Download {i}: {res}')
