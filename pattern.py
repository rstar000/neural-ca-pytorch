import urllib
import torch
import PIL
import numpy as np


def load_pattern(filename, max_size=None):
    img = PIL.Image.open(filename)
    if max_size is not None:
        img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)

    img_np = np.array(img)
    # img_np[..., :3] *= img_np[..., 3][..., None]
    return img_np


def load_emoji_pattern(emoji, max_size):
    code = hex(ord(emoji))[2:].lower()
    url = 'https://github.com/googlefonts/noto-emoji/raw/master/png/128/emoji_u%s.png' % code

    urllib.request.urlretrieve(url, 'image.png')
    return load_pattern('image.png', max_size)


def put_on_grid(pattern, grid):
    """
    Args:
        pattern (np.array):
            array of shape (H, W, C) with RGB uint8 values
        grid (torch.Tensor):
            State grid of shape C, H, W
    """

    pattern = pattern.astype(np.float32) / 255.0
    pattern = pattern.transpose(2, 0, 1)  # to CHW
    pattern = torch.from_numpy(pattern)

    pc, ph, pw = pattern.shape
    gc, gh, gw = grid.shape

    assert pc == 4, "Pattern must be RGBA"
    assert ph <= gh and pw <= gw

    offset_x = (gw - pw) // 2
    offset_y = (gh - ph) // 2

    grid[:4, offset_y:offset_y + ph, offset_x:offset_x+pw] = pattern
    return grid


def tensor_to_img(x):
    img = x.detach().numpy()
    img = img.transpose(1, 2, 0)
    img = img[:,:,:4]
    img = img.clip(0, 1)
    img = img * 255
    img = img.astype(np.uint8)
    return img
