from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow import keras
import tensorflow as tf
import cv2
import math
import numpy as np
import torch
from torch.nn import functional as F


def downsample(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization())

    result.add(layers.PReLU())

    return result


def upsample(filters, size, apply_batch=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    if apply_batch:
        # tfa.layers.InstanceNormalization())
        result.add(layers.BatchNormalization())

    result.add(layers.PReLU())

    return result


class self_attention(tf.keras.Model):
    def __init__(self, channels):
        super(self_attention, self).__init__(name='')
        self.channels = channels
        self.f = layers.Conv2D(channels // 8, kernel_size=1,
                               strides=1, )  # [bs, h, w, c']
        self.g = layers.Conv2D(channels // 8, kernel_size=1,
                               strides=1, )  # [bs, h, w, c']
        self.h = layers.Conv2D(channels // 2, kernel_size=1,
                               strides=1, )  # [bs, h, w, c]
        self.last_ = layers.Conv2D(
            self.channels, kernel_size=1, strides=1, activation='relu')

        self.dropout = tf.keras.layers.Dropout(0.1)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def hw_flatten(self, x):
        # layers.Reshape(( -1, x.shape[-2], x.shape[-1]))(x)
        return layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    def reshape(self, x, height, width, num_channels):
        return layers.Reshape((height, width, num_channels//2))(x)

    def call(self, x):
        batch_size, height, width, num_channels = x.get_shape().as_list()

        f = self.f(x)
        g = self.g(x)
        h = self.h(x)
        dk = tf.cast(tf.shape(g)[-1], tf.float32)

        # N = h * w
        s = tf.matmul(self.hw_flatten(g), self.hw_flatten(
            f), transpose_b=True)/tf.math.sqrt(dk)  # [bs, N, N]
        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, self.hw_flatten(h), transpose_a=True)  # [bs, N, C]
        # tf.compat.v1.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        gamma = 0.002
        o = self.reshape(o, height, width, num_channels)  # [bs, h, w, C]
        o = self.last_(o)
        out = self.dropout(o)
        out = self.layernorm(gamma * out + x)

        return out


class self_attentionDecoder(tf.keras.Model):
    def __init__(self, channels):
        super(self_attentionDecoder, self).__init__(name='')
        self.channels = channels
        self.f = layers.Conv2DTranspose(
            channels // 8, kernel_size=1, strides=1, )  # [bs, h, w, c']
        self.g = layers.Conv2DTranspose(
            channels // 8, kernel_size=1, strides=1, )  # [bs, h, w, c']
        self.h = layers.Conv2DTranspose(
            channels // 2, kernel_size=1, strides=1, )  # [bs, h, w, c]
        self.last_ = layers.Conv2DTranspose(
            self.channels, kernel_size=1, strides=1, activation='relu')

        self.dropout = tf.keras.layers.Dropout(0.1)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def hw_flatten(self, x):
        # layers.Reshape(( -1, x.shape[-2], x.shape[-1]))(x)
        return layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    def reshape(self, x, height, width, num_channels):
        return layers.Reshape((height, width, num_channels//2))(x)

    def call(self, x):
        batch_size, height, width, num_channels = x.get_shape().as_list()

        f = self.f(x)
        g = self.g(x)
        h = self.h(x)
        dk = tf.cast(tf.shape(g)[-1], tf.float32)

        # N = h * w
        s = tf.matmul(self.hw_flatten(g), self.hw_flatten(
            f), transpose_b=True)/tf.math.sqrt(dk)  # [bs, N, N]
        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, self.hw_flatten(h))  # [bs, N, C]
        # tf.compat.v1.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        gamma = 0.002
        o = self.reshape(o, height, width, num_channels)  # [bs, h, w, C]
        o = self.last_(o)
        out = self.dropout(o)
        out = self.layernorm(gamma * out + x)

        return out


def Generator(HEIGHT, WIDTH):
    inputs = layers.Input(shape=[HEIGHT, WIDTH, 3])
    OUTPUT_CHANNELS = 3
    # bs = batch size
    down_stack = [
        downsample(64, 4, apply_instancenorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)

        # downsample(512, 4),  # (bs, 16, 16, 512)
        # downsample(512, 4)  # (bs, 16, 16, 512)
    ]

    up_stack = [
        # upsample(512, 4, True),  # (bs, 16, 16, 1024)
        # upsample(512, 4, True),  # (bs, 16, 16, 1024)

        upsample(512, 4, True),  # (bs, 16, 16, 1024)
        upsample(256, 4, True),  # (bs, 32, 32, 512)
        upsample(128, 4, True),  # (bs, 64, 64, 256)
        upsample(64, 4, True)  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 3,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh')  # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    # Bottleneck
    attention0 = self_attention(512)
    x = attention0(x)

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])
    attention1 = self_attentionDecoder(192)
    x = attention1(x)
    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)


class RealESRGANer():
    """A helper class for upsampling images with RealESRGAN.
    Args:
        scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
        model_path (str): The path to the pretrained model. It can be urls (will first download it automatically).
        model (nn.Module): The defined network. Default: None.
        tile (int): As too large images result in the out of GPU memory issue, so this tile option will first crop
            input images into tiles, and then process each of them. Finally, they will be merged into one image.
            0 denotes for do not use tile. Default: 0.
        tile_pad (int): The pad size for each tile, to remove border artifacts. Default: 10.
        pre_pad (int): Pad the input images to avoid border artifacts. Default: 10.
        half (float): Whether to use half precision during inference. Default: False.
    """

    def __init__(self, scale, model_path, model=None, tile=0, tile_pad=10, pre_pad=10, half=False, device=None):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half

        # initialize model
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        # if the model_path starts with https, it will first download models to the folder: realesrgan/weights
        loadnet = torch.load(model_path, map_location=torch.device('cpu'))
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)
        model.eval()
        self.model = model.to(self.device)
        if self.half:
            self.model = self.model.half()

    def pre_process(self, img):
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible
        """
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
        if self.half:
            self.img = self.img.half()

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad,
                             0, self.pre_pad), 'reflect')
        # mod pad for divisible borders
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = F.pad(self.img, (0, self.mod_pad_w,
                             0, self.mod_pad_h), 'reflect')

    def process(self):
        # model inference
        self.output = self.model(self.img)

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad,
                                      input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    with torch.no_grad():
                        output_tile = self.model(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (
                    input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (
                    input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    def post_process(self):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h *
                                      self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad *
                                      self.scale, 0:w - self.pre_pad * self.scale]
        return self.output

    @torch.no_grad()
    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        h_input, w_input = img.shape[0:2]
        # img: numpy
        img = img.astype(np.float32)
        if np.max(img) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        img = img / max_range
        if len(img.shape) == 2:  # gray image
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha_upsampler == 'realesrgan':
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ------------------- process image (without the alpha channel) ------------------- #
        self.pre_process(img)
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()
        output_img = self.post_process()
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # ------------------- process the alpha channel if necessary ------------------- #
        if img_mode == 'RGBA':
            if alpha_upsampler == 'realesrgan':
                self.pre_process(alpha)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_alpha = self.post_process()
                output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(
                    output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:  # use the cv2 resize for alpha channel
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(
                    alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)

            # merge the alpha channel
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # ------------------------------ return ------------------------------ #
        if max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output, (
                    int(w_input * outscale),
                    int(h_input * outscale),
                ), interpolation=cv2.INTER_LANCZOS4)

        return output, img_mode
