import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from random import random, choice, randint
from scipy.ndimage.filters import gaussian_filter  # keep for compatibility
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def data_augment(img, opt):
    """
    Apply in-place Gaussian blur and optional JPEG compression based on probabilities in `opt`.
    - Expects a PIL Image; returns a PIL Image.
    - Uses `opt.blur_prob`, `opt.blur_sig`, `opt.jpg_prob`, `opt.jpg_method`, `opt.jpg_qual`.
    """
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    """
    Sample a continuous value:
    - If `s` has length 1, return s[0].
    - If length 2, sample uniformly in [s[0], s[1]].
    """
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    """
    Sample a discrete value:
    - If `s` has length 1, return s[0]; otherwise pick a random element.
    """
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur_gray(img, sigma):
    """
    Apply Gaussian blur on grayscale or each channel for color images; returns a new array.
    """
    if len(img.shape) == 3:
        img_blur = np.zeros_like(img)
        for i in range(img.shape[2]):
            img_blur[:, :, i] = gaussian_filter(img[:, :, i], sigma=sigma)
    else:
        img_blur = gaussian_filter(img, sigma=sigma)
    return img_blur


def gaussian_blur(img, sigma):
    """
    In-place Gaussian blur on each of the 3 channels of `img` (H, W, 3).
    """
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def add_noise(img, sigma):
    """
    Additive Gaussian noise with std `sigma` to uint8 image `img`; returns uint8 array.
    """
    noise = np.random.normal(0, sigma, img.shape)
    norm_img = img / 255.0
    norm_img += noise
    return (norm_img * 255).astype("uint8")


def random_crop(img, ratio, resize=True):
    """
    Randomly crop a subregion with side ratio `ratio` and resize back to original size.
    Uses INTER_NEAREST for resizing to match original behavior.
    """
    h, w = img.shape[:2]
    th, tw = int(h * ratio), int(w * ratio)
    i = randint(0, h - th)
    j = randint(0, w - tw)
    cropped = img[i: i + th, j: j + tw]
    return cv2.resize(cropped, (h, w), interpolation=cv2.INTER_NEAREST)


def cv2_jpg_gray(img, compress_val):
    """
    JPEG encode/decode a grayscale image using OpenCV with quality `compress_val`.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    _, encimg = cv2.imencode(".jpg", img, encode_param)
    decimg = cv2.imdecode(encimg, 0)
    return decimg


def cv2_jpg(img, compress_val):
    """
    JPEG encode/decode a color image (BGR<->RGB handled) using OpenCV with quality `compress_val`.
    """
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    _, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    """
    JPEG encode/decode a color image using PIL with quality `compress_val`.
    """
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {"cv2": cv2_jpg, "pil": pil_jpg}


def jpeg_from_key(img, compress_val, key):
    """
    Dispatch JPEG encode/decode by `key` in {'cv2', 'pil'} with quality `compress_val`.
    """
    method = jpeg_dict[key]
    return method(img, compress_val)


def get_processing_model(opt):
    """
    Configure processing model options based on `opt.detect_method`.
    Currently supports: 'D3QE' only.
    """
    print("Processing model: ", opt.detect_method)
    if opt.detect_method == "D3QE":
        opt.norm_type = "vae"
        opt.CropSize = 256
    else:
        raise ValueError(f"Unsupported model_type: {opt.detect_method}")
    return opt


rz_dict = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "nearest": Image.NEAREST,
}


def custom_resize(img, opt):
    """
    Resize `img` to `opt.loadSize` using a randomly sampled interpolation from `opt.rz_interp`.
    """
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])


def processing(img, opt, name):
    """
    Build and apply the preprocessing pipeline:
    - Resize (optional), augmentation (train/val), crop, optional flip, to tensor, normalize by `name`.
    - `name` in {'imagenet', 'clip', 'vae'} or None (no normalization).
    - Returns a tensor.
    """
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.CropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.CropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)

    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))

    trans = transforms.Compose(
        [
            rz_func,
            transforms.Lambda(
                lambda img: (data_augment(img, opt) if (opt.isTrain or opt.isVal) else img)
            ),
            crop_func,
            flip_func,
            transforms.ToTensor(),
            (
                transforms.Normalize(mean=MEAN[name], std=STD[name])
                if name is not None
                else transforms.Lambda(lambda img: img)
            ),
        ]
    )
    return trans(img)


MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073],
    "vae": [0.5, 0.5, 0.5],
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711],
    "vae": [0.5, 0.5, 0.5],
}


def normlize_np(img):
    """
    Normalize a numpy array to [0, 255] range; returns float array.
    """
    img -= img.min()
    if img.max() != 0:
        img /= img.max()
    return img * 255.0


processimg = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)