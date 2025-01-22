import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# def create_lpf_rect(N, cutoff=0.5):
#     cutoff_low = int((N * cutoff) // 2)
#     cutoff_high = int(N - cutoff_low)
#     rect_1d = torch.ones(N)
#     rect_1d[cutoff_low + 1:cutoff_high] = 0
#     if N % 4 == 0:
#         # if N is divides by 4, nyquist freq should be 0
#         # N % 4 =0 means the downsampeled signal is even
#         rect_1d[cutoff_low] = 0
#         rect_1d[cutoff_high] = 0
#
#     rect_2d = rect_1d[:, None] * rect_1d[None, :]
#     return rect_2d

# support not squared images
def create_lpf_rect(N1, N2, cutoff=0.5):
    rects = []
    for N in [N1, N2]:
        cutoff_low = int((N * cutoff) // 2)
        cutoff_high = int(N - cutoff_low)
        rect_1d = torch.ones(N)
        rect_1d[cutoff_low + 1:cutoff_high] = 0
        if N % 4 == 0:
            # if N is divides by 4, nyquist freq should be 0
            # N % 4 =0 means the downsampeled signal is even
            rect_1d[cutoff_low] = 0
            rect_1d[cutoff_high] = 0
        rects.append(rect_1d)
    rect_2d = rects[0][:, None] * rects[1][None, :]
    return rect_2d

# upsample using FFT
# def create_recon_rect(N, cutoff=0.5):
#     cutoff_low = int((N * cutoff) // 2)
#     cutoff_high = int(N - cutoff_low)
#     rect_1d = torch.ones(N)
#     rect_1d[cutoff_low + 1:cutoff_high] = 0
#     if N % 4 == 0:
#         # if N is divides by 4, nyquist freq should be 0.5
#         # N % 4 =0 means the downsampeled signal is even
#         rect_1d[cutoff_low] = 0.5
#         rect_1d[cutoff_high] = 0.5
#     rect_2d = rect_1d[:, None] * rect_1d[None, :]
#     return rect_2d

def create_recon_rect(N1, N2, cutoff=0.5):
    rects = []
    for N in [N1, N2]:
        cutoff_low = int((N * cutoff) // 2)
        cutoff_high = int(N - cutoff_low)
        rect_1d = torch.ones(N)
        rect_1d[cutoff_low + 1:cutoff_high] = 0
        if N % 4 == 0:
            rect_1d[cutoff_low] = 0.5
            rect_1d[cutoff_high] = 0.5
        rects.append(rect_1d)
    rect_2d = rects[0][:, None] * rects[1][None, :]
    return rect_2d


class LPF_RFFT(nn.Module):
    '''
        saves rect in first use
        '''
    def __init__(self, cutoff=0.5):
        super(LPF_RFFT, self).__init__()
        self.cutoff = cutoff

    def forward(self, x):
        x_fft = torch.fft.rfft2(x)
        N1, N2 = x.shape[-2], x.shape[-1]
        rect = create_lpf_rect(N1, N2, self.cutoff)[:,:int(N2/2+1)].to(x.device)
        x_fft *= rect
        out = torch.fft.irfft2(x_fft)
        return out


class LPF_RECON_RFFT(nn.Module):
    '''
        saves rect in first use
        '''
    def __init__(self, cutoff=0.5):
        super(LPF_RECON_RFFT, self).__init__()
        self.cutoff = cutoff

    def forward(self, x):
        x_fft = torch.fft.rfft2(x)
        # if not hasattr(self, 'rect'):
        #     N = x.shape[-1]
            # cannot register rect because x size changes
            # self.register_buffer('rect', create_recon_rect(N, self.cutoff)[:,:int(N/2+1)])
            # self.to(x.device)
        # x_fft *= self.rect
        N1, N2 = x.shape[-2], x.shape[-1]
        rect = create_recon_rect(N1, N2, self.cutoff)[:,:int(N2/2+1)].to(x.device)
        x_fft *= rect
        out = torch.fft.irfft2(x_fft)
        return out


class UpsampleRFFT(nn.Module):
    '''
    input shape is unknown
    '''
    def __init__(self, up=2):
        super(UpsampleRFFT, self).__init__()
        self.up = up
        self.recon_filter = LPF_RECON_RFFT(cutoff=1 / up)

    def forward(self, x):
        # pad zeros
        batch_size, num_channels, in_height, in_width = x.shape
        x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
        x = torch.nn.functional.pad(x, [0, self.up - 1, 0, 0, 0, self.up - 1])
        x = x.reshape([batch_size, num_channels, in_height * self.up, in_width * self.up])
        x = self.recon_filter(x) * (self.up ** 2)
        return x


class UpsampleRFFT2(nn.Module):
    '''
    Upsampling in Fourier domain, including fractional upsample
    '''
    def __init__(self, up=2):
        super(UpsampleRFFT2, self).__init__()
        self.up = up

    def forward(self, x):
        x = torch.fft.rfft2(x)
        y_shape = list(x.shape)
        rows_low = int(y_shape[2] / 2) + 1
        rows_high = int(y_shape[2] / 2)
        cols = y_shape[3]

        y_shape[2] = math.ceil(self.up * y_shape[2])
        y_shape[3] = int(y_shape[2] / 2) + 1

        # y = torch.zeros(y_shape, dtype=x.dtype).to(x.device)
        y = torch.zeros(y_shape, dtype=x.dtype, device=x.device)

        # y[:,:,0:rows_low, 0:cols] = x[:,:,0:rows_low, 0:cols] * (self.up ** 2)
        # y[:, :, -rows_high:, 0:cols] = x[:, :, -rows_high:, 0:cols] * (self.up ** 2)

        y[:,:,0:rows_low, 0:cols] = x[:,:,0:rows_low, 0:cols]
        y[:, :, -rows_high:, 0:cols] = x[:, :, -rows_high:, 0:cols]

        if x.shape[2] % 2 == 0:
            # if N is divides by 2, nyquist freq should be 0.5
            # N % 2 = 0 means signal size is evene
            y[:,:,:,cols-1] *= 0.5
            y[:,:, rows_low-1, :] *= 0.5
            y[:,:, -rows_high, :] *= 0.5

        y *= self.up ** 2

        y = torch.fft.irfft2(y)
        return y




def test_upsample_rfft2():
    from blurpool import BlurPool
    print("test_upsample_rfft2")
    N = 5
    up = UpsampleRFFT(up=2)
    up2 = UpsampleRFFT2(up=2)

    x = torch.randn(1,1,N,N)

    y = up(x)
    y2 = up2(x)

    y_rfft = torch.fft.rfft2(y)
    y2_rfft = torch.fft.rfft2(y2)

    plt.figure()
    plt.imshow(torch.abs(y_rfft)[0,0])
    plt.title('y_rfft')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(torch.abs(y2_rfft)[0, 0])
    plt.title('y2_rfft')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(torch.abs(y2_rfft - y_rfft)[0, 0])
    plt.title('diff')
    plt.colorbar()
    plt.show()

    # upsample - down_sample

    # up x2 vs up x4 -> down x2
    up2 = UpsampleRFFT2(up=2)

    up4down2 = nn.Sequential(UpsampleRFFT(up=4),
                             BlurPool(channels=1, pad_type='circular', filter_type='ideal'))

    x = torch.randn(1, 1, N, N)

    y1 = up2(x)
    y2 = up4down2(x)
    diff = torch.sum(torch.abs(y1 - y2))
    print("up x2 vs up x4 -> down x2 diff:", diff)

    # up x1.5 vs up x3 -> down x2
    up1_5 = UpsampleRFFT2(up=1.5)

    up3down2 = nn.Sequential(UpsampleRFFT(up=3),
                             BlurPool(channels=1, pad_type='circular', filter_type='ideal'))

    x = torch.randn(1, 1, N, N)

    y1 = up1_5(x)
    y2 = up3down2(x)
    diff = torch.sum(torch.abs(y1 - y2))
    print("up x1.5 vs up x3 -> down x2 diff:", diff)


    print()







def test_upsample_rfft():
    print("test_upsample_rfft")
    up = UpsampleRFFT(up=2)
    inp = torch.randn(1,1,9,9)
    inp_rfft = torch.fft.rfft2(inp)

    out = up(inp)
    out_rfft = torch.fft.rfft2(out)

    plt.figure()
    plt.imshow(inp_rfft[0,0].real)
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(out_rfft[0,0].real)
    plt.colorbar()
    plt.show()
    print()


def upsample_partial_test():
    # check how zero padding in space effects rfft:
    N = 4
    x = torch.randn(1,1,N,N)
    x_rfft = torch.fft.rfft2(x)

    # zero padding
    up = 2
    batch_size, num_channels, in_height, in_width = x.shape
    x_pad = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x_pad = torch.nn.functional.pad(x_pad, [0, up - 1, 0, 0, 0, up - 1])
    x_pad = x_pad.reshape([batch_size, num_channels, in_height * up, in_width * up])

    x_pad_rfft = torch.fft.rfft2(x_pad)

    x_up2 = UpsampleRFFT(2)(x)
    x_up2_rfft = torch.fft.rfft2(x_up2)

    plt.imshow(torch.abs(x_rfft)[0,0])
    plt.title('x_rfft')
    plt.colorbar()
    plt.show()
    plt.imshow(torch.abs(x_pad_rfft)[0, 0])
    plt.title('x_pad_rfft')
    plt.colorbar()
    plt.show()
    plt.imshow(torch.abs(x_up2_rfft)[0, 0])
    plt.title('x_up2_rfft')
    plt.colorbar()
    plt.show()

    print()

def subpixel_shift(images, up=2, shift_x=1, shift_y=1, up_method='ideal'):
    '''
    effective fractional shift is (shift_x / up, shift_y / up)
    '''

    assert up_method == 'ideal', 'Only "ideal" interpolation kenrel is supported'
    up_layer = UpsampleRFFT(up=up).to(images.device)
    up_img_batch = up_layer(images)
    # img_batch_1 = up_img_batch[:, :, 1::2, 1::2]
    img_batch_1 = torch.roll(up_img_batch, shifts=(-shift_x, -shift_y), dims=(2, 3))[:, :, ::up, ::up]
    return img_batch_1


if __name__ == '__main__':
    test_upsample_rfft()
    # upsample_partial_test()
    test_upsample_rfft2()
