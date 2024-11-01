from PIL import Image, ImageFont, ImageDraw, ImageColor
import pandas as pd
from tqdm import tqdm


def text_to_image(
    text: str,
    font,
    color: (int, int, int), #color is in RGB
    ):

    mask_image = font.getmask(text, "L")
    img = Image.new("RGB", mask_image.size)
    img.im.paste(color, (0, 0) + mask_image.size, mask_image)  # need to use the inner `img.im.paste` due to `getmask` returning a core   return img

    return img

def random_string(length: int):
    import random
    import string

    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length))

def main(
    dataset_rootdir='dataset2',
    string_length=10,
    number_of_samples=100000
    ):
    # font_filepath = "/Library/Fonts/Arial Unicode.ttf"
    font_filepath = '/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf'

    # text = "Hello, World 12!"
    font_size = 32
    color = (255, 255, 255)
    font = ImageFont.truetype(font_filepath, size=font_size)


    df = [{
        'text': random_string(string_length),
        'image': f'imgs/{i}.png'
    } for i in range(number_of_samples)]

    pd.DataFrame(df).to_csv(f"{dataset_rootdir}/data.csv", index=False)

    for text in tqdm(df):
        img = text_to_image(text['text'], font, color)
        img.save(f'{dataset_rootdir}/' + text['image'])

if __name__ == '__main__':
    main()