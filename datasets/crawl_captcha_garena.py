import random

def gen_captcha_key() -> str:
    raw = list('xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx')
    for i, c in enumerate(raw):
        if c == 'x' or c == 'y':
            r = random.randint(0, 16) | 0
            if c == 'x':
                 v = r
            else:
                v = r & 0x3 | 0x8
            raw[i] = hex(v)[2:]

    result = ''.join(raw).replace('-', '')
    return result


url = f'https://gop.captcha.garena.com/image?key={gen_captcha_key()}'
