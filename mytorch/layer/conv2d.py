from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.util import initializer

import numpy as np

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), need_bias: bool = False, mode="xavier") -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        batch_size, in_channels, in_height, in_width  = x.shape

        out_height = ((in_height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0]) + 1
        out_width = ((in_width  + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1]) + 1

        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        if self.padding[0] > 0 or self.padding[1] > 0:
            x_padded  = np.pad(x.data, ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), mode='constant')
        else:
            x_padded = x.data

        for i in range(out_height):
            h_start = i * self.stride[0]
            h_end = h_start + self.kernel_size[0]
            for j in range(out_width):
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]
                x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]
                for c_o in range(self.out_channels):
                    output[:, c_o, i, j] = np.sum((x_slice * self.weight.data[c_o, :, :, :]), axis=(1, 2, 3))

        if self.need_bias:
            output += self.bias.data.reshape((1, self.out_channels, 1, 1))

        return Tensor(output)
    
    def initialize(self):
        "TODO: initialize weights"
        self.weight = Tensor(
            data=initializer(shape=(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]), mode=self.initialize_mode),
            requires_grad=True
        )
        if self.need_bias:
            self.bias = Tensor(
                data=initializer((self.out_channels, 1), "zero"),
                requires_grad=True
            )

    def zero_grad(self):
        "TODO: implement zero grad"
        self.weight.zero_grad()
        if self.need_bias:
            self.bias.zero_grad()

    def parameters(self):
        "TODO: return weights and bias"
        if self.need_bias:
            return [self.weight, self.bias]
        return [self.weight]
    
    def __str__(self) -> str:
        return "conv 2d - total params: {} - kernel: {}, stride: {}, padding: {}".format(
                                                                                    self.kernel_size[0] * self.kernel_size[1],
                                                                                    self.kernel_size,
                                                                                    self.stride, self.padding)
