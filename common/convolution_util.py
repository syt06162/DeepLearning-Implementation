import numpy as np

# input - output size
def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return int((input_size + 2*pad - filter_size) / stride + 1)

# im2col
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = conv_output_size(H, filter_h, stride, pad)
    out_w = conv_output_size(W, filter_w, stride, pad)

    # padding image (h and w padding)
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant') # 4dim
    
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w)) # 6 dim

    for y in range(0, filter_h):
        y_max = y + stride*out_h
        for x in range(0, filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

# col2im
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = conv_output_size(H, filter_h, stride, pad)
    out_w = conv_output_size(W, filter_w, stride, pad)

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
