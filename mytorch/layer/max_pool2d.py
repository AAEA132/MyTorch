from mytorch import Tensor
from mytorch.layer import Layer

import numpy as np

class MaxPool2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)) -> None:
        super()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        batch_size, channels, in_height, in_width = x.data.shape
        
        out_height = (in_height - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        out_width = (in_width - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1

        output = np.zeros((batch_size, channels, out_height, out_width))

        for b in range(batch_size):
            for c in range(channels):
                for h in range(0, in_height - self.kernel_size[0] + 1, self.stride[0]):
                    for w in range(0, in_width - self.kernel_size[1] + 1, self.stride[1]):
                        output[b, c, h//self.stride[0], w//self.stride[1]] = np.max(x.data[b, c, h:h+self.kernel_size[0], w:w+self.kernel_size[1]])

        return Tensor(output)
    
    def __str__(self) -> str:
        return "max pool 2d - kernel: {}, stride: {}, padding: {}".format(self.kernel_size, self.stride, self.padding)
