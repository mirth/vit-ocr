import random
from PIL import Image, ImageFont, ImageDraw, ImageColor
import pandas as pd
from tqdm import tqdm
import os
import glob
from pathlib import Path


def text_to_image(text, font):
    image = Image.new("RGB", (224, 224), random_color())
    draw = ImageDraw.Draw(image)
    
    draw.text(
        random_pos(),
        text,
        random_color(),
        font=font,
    )


    return image

def text_to_image0(
    text: str,
    font,
    background_color,
    ):
    mask_image = font.getmask(text, "L")
    img = Image.new("RGB", mask_image.size, background_color)
    img.im.paste(random_color(), (0, 0) + mask_image.size, mask_image)  # need to use the inner `img.im.paste` due to `getmask` returning a core   return img

    return img

def random_pos():
    return (
        random.randint(0, 20),
        random.randint(0, 180),
    )



def random_string(length):
    import random
    import string

    length = random.randint(*length)

    return ''.join(random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits) for _ in range(length))

def random_color():
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )

def random_font_size():
    return random.randint(32, 32)

def main(
    dataset_rootdir='datasets/dataset8',
    string_length=(1, 22),
    number_of_samples_per_font=1000000
    ):
    fonts_dir = Path('/home/tolik/Downloads/Fonts')
    fonts = glob.glob(str(fonts_dir / '*.ttf')) + glob.glob(str(fonts_dir / '*.TTF'))

    df = [{
        'text': random_string(string_length),
        'image': f'imgs/{i}.png',
        'font': random.choice(fonts),
    } for i in range(number_of_samples_per_font)]

    pd.DataFrame(df).to_csv(f"{dataset_rootdir}/data.csv", index=False)

    for row in tqdm(df):
        font_size = 32#random_font_size()
        try:
            font = ImageFont.truetype(fonts_dir / row['font'], size=font_size)
            img = text_to_image0(row['text'], font, background_color=random_color())
            img.save(f'{dataset_rootdir}/' + row['image'])
        except Exception as e:
            print(e)

if __name__ == '__main__':
    main()