import random
from PIL import Image, ImageFont, ImageDraw, ImageColor
import pandas as pd
from tqdm import tqdm


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

    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length))

def random_color():
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )

def random_font_size():
    return random.randint(12, 32)

def main(
    dataset_rootdir='datasets/dataset7',
    string_length=(10, 10),
    number_of_samples=100000
    ):
    # font_filepath = "/Library/Fonts/Arial Unicode.ttf"
    font_filepath = '/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf'

    df = [{
        'text': random_string(string_length),
        'image': f'imgs/{i}.png'
    } for i in range(number_of_samples)]

    pd.DataFrame(df).to_csv(f"{dataset_rootdir}/data.csv", index=False)

    for text in tqdm(df):
        font_size = 32#random_font_size()
        font = ImageFont.truetype(font_filepath, size=font_size)

        img = text_to_image0(text['text'], font, background_color=(0, 0, 0))
        img.save(f'{dataset_rootdir}/' + text['image'])

if __name__ == '__main__':
    main()