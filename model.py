'''
PyTorch implementation of Deep-CapsNet architecture.
'''

import torch
import torch.nn as nn


def squash(caps, dim=-1, eps=1e-8):
    '''
    CapsNet's non-linear activation function for capsules.
    '''
    dot_prod = torch.sum(caps**2, axis=dim, keepdim=True) #dot product
    scalar_factor = dot_prod/(1+dot_prod)/torch.sqrt(dot_prod + eps)
    squashed = scalar_factor * caps
    return squashed


class ConvertToCaps(nn.Module):
    '''
    Converts the given conv outputs to capsules.
    '''
    def __init__(self):
        super(self, ConvertToCaps).__init__()

    def forward(self, x):
        '''
        Adds a dimension for the capsules in the conv output. In the original paper, squash activation function was applied right after the dimension
        expansion took place. However, in the official implementation, no squashing was applied. Here we'll try both the implementation and see what comes
        on top. The activation function is what ensures the discriminative learning to treat the tensors as capsules.
        '''
        return squash(torch.unsqueeze(x, dim=2), dim=2)




class Conv2DCaps(nn.Module):
    '''
    2D Convolution on capsules.
    '''

    def __init__(self, height, width, conv_channel_in, caps_num_in, conv_channel_out, caps_num_out, kernel_size=3, stride=1, routing_iter=1, pad=1):
        '''
        Parameter Init.
        '''

        super(self, Conv2DCaps).__init__()
        self.height = height
        self.width = width
        self.conv_channel_in = conv_channel_in
        self.caps_num_in = caps_num_in
        self.conv_channel_out = conv_channel_out
        self.caps_num_out = caps_num_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.routing_iter = routing_iter
        self.pad = pad

        #Capsule 2D convolution works by temporarily reshaping the capsule tensors[batch size, channels, num capsules, feature width, feature height]
        #back to conv tensors [batch size, channels, feature width, feature height] and perform a typical 2D Convolution process. The output of this
        #process is then converted back to capsule tensors. The final capsule tensors will undergo the squash activation function before being returned.

        #The given capsule tensor inputs [batch size, channels, num capsules, feature width, feature height] is reshaped into conv tensors by
        #treating the channels*numcapsules as the new channel input. The output of the conv operation would be also treated as such which enables
        #the final output channels to be reshaped back into the desired num of capsules.
        reshaped_in_channels = self.conv_channel_in*self.caps_num_in
        reshaped_out_channels= self.conv_channel_out*self.caps_num_out

        self.conv = nn.Conv2d(in_channels=reshaped_in_channels, out_channels=reshaped_out_channels, kernel_size=self.kernel_size, stride=self.stride,
                            pad=self.pad)


    def forward(self, inputs):
        '''
        Forward Propagation.
        '''

        batch_size = inputs.size()[0]
        #reshape the capsule tensor to the conv tensor as explained above.
        caps_reshaped = inputs.view(batch_size, self.conv_channel_in*self.caps_num_in, self.height, self.width)

        conv_output = self.conv(caps_reshaped) #conv process.

        height,width = conv_output.size()[-2:] #the size of the feature map after applying convolution.

        conv_reshaped = conv_output.view(batch_size, self.conv_channel_out, self.caps_num_out, height, width) #reshape the conv output to capsules.

        return squash(conv_reshaped, dim=2) #apply the activation function before returning the capsules.



class Conv3DCaps(nn.Module):
    '''
    3D Convolution on capsules.
    '''

    def __init__(self, height, width, conv_channel_in, caps_num_in, conv_channel_out, caps_num_out, kernel_size=3, routing_iter=3):
        '''
        Parameter Init.
        '''
        super(self, Conv3DCaps).__init__()

        self.height = height
        self.width = width
        self.conv_channel_in = conv_channel_in
        self.caps_num_in = caps_num_in
        self.conv_channel_out = conv_channel_out
        self.caps_num_out = caps_num_out
        self.kernel_size = kernel_size
        self.routing_iter = routing_iter

        reshaped_in_channels = 1
        reshaped_out_channels = self.caps_num_out * self.conv_channel_out
        stride = (caps_num_in, 1, 1)
        pad = (0, 1, 1)








class DeepCaps(nn.Module):
    '''
    DeepCaps Model.
    '''

    def __init__(self):
        '''
        Init the architecture and parameters.
        '''

        super(self, DeepCaps).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(num_features=128, eps=1e-08, momentum=0.99)

        self.toCaps = ConvertToCaps()

        self.conv2dcaps_00 = Conv2DCaps(height=28, width=28, conv_channel_in=128, caps_num_in=1, conv_channel_out=32,
                                        caps_num_out=4, stride=2)
        self.conv2dcaps_01 = Conv2DCaps(height=14, width=14, conv_channel_in=32, caps_num_in=4, conv_channel_out=32,
                                        caps_num_out=4, stride=1)
        self.conv2dcaps_02 = Conv2DCaps(height=14, width=14, conv_channel_in=32, caps_num_in=4, conv_channel_out=32,
                                        caps_num_out=4, stride=1)
        self.conv2dcaps_03 = Conv2DCaps(height=14, width=14, conv_channel_in=32, caps_num_in=4, conv_channel_out=32,
                                        caps_num_out=4, stride=1)


        self.conv2dcaps_10 = Conv2DCaps(height=14, width=14, conv_channel_in=32, caps_num_in=4, conv_channel_out=32,
                                        caps_num_out=8, stride=2)
        self.conv2dcaps_11 = Conv2DCaps(height=7, width=7, conv_channel_in=32, caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=1)
        self.conv2dcaps_12 = Conv2DCaps(height=7, width=7, conv_channel_in=32, caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=1)
        self.conv2dcaps_13 = Conv2DCaps(height=7, width=7, conv_channel_in=32, caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=1)
        self.conv2dcaps_13 = Conv2DCaps(height=7, width=7, conv_channel_in=32, caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=1)


        self.conv2dcaps_20 = Conv2DCaps(height=7, width=7, conv_channel_in=32, caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=2)
        self.conv2dcaps_21 = Conv2DCaps(height=4, width=4, conv_channel_in=32, caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=1)
        self.conv2dcaps_22 = Conv2DCaps(height=4, width=4, conv_channel_in=32, caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=1)
        self.conv2dcaps_23 = Conv2DCaps(height=4, width=4, conv_channel_in=32, caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=1)


        self.conv2dcaps_30 = Conv2DCaps(height=4, width=4, conv_channel_in=32, caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=2)
        self.conv3dcaps_31 = Conv3DCaps()




