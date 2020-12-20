'''
PyTorch implementation of Deep-CapsNet architecture.
'''

import torch
import torch.nn as nn
import torch.nn.functional as func


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

        #In the original CapsNet architecture (2017), every capsule in Primary Layer will be multiplied by a (possibly unique) transformation matrices
        #(i.e. fully connected) and then each of those transformed capsules will be used to vote for every capsule in the DigitCaps layer. In this paper,
        #a SUBSET of capsule in a block will be transformed to be used to vote for the higher level capsules (3D Dynamic Routing). NOTE: the stride of
        #'caps_num_in' during 3D Conv makes sure only a subset of one single capsule will be transformed and that the number of features outputted can be
        #converted to the number of desired output capsules. In contrast to the original dynamic routing, 3D dynamic routing have far less parameters.
        #Since all the capsules are products of convolution operations, adjacent capsules would contain similar information. Thus the stride.

        reshaped_in_channels = 1
        reshaped_out_channels = self.caps_num_out * self.conv_channel_out
        stride = (caps_num_in, 1, 1)
        pad = (0, 1, 1)

        self.3dconv = nn.Conv3d(in_channels=reshaped_in_channels,
                                out_channels=reshaped_out_channels,
                                kernel_size=self.kernel_size,
                                padding=pad)


        def forward(self, x):
            '''
            Forward Propagation.
            '''
            batch_size = x.size()[0]
            x = x.view(batch_size, self.conv_channel_in*self.caps_num_in, self.height, self.width)
            x = x.unsqueeze(1)
            x = self.3dconv(x)
            self.height, self.width = x.size()[-2:]

            #ALL the permute operations are done according to the paper.
            x = x.permute(0,2,1,3,4)
            x = x.view(batch_size, self.conv_channel_in, self.conv_channel_out, self.caps_num_out, self.height, self.width)

            x = x.permute(0, 4, 6, 3, 2, 1).contiguous()
            self.B = x.new(batch_size, self.width, self.height, 1, self.caps_num_out, self.caps_num_in).zero()

            x = self.routing(x, batch_size, self.routing_iter)



        def routing(self, x, batch_size, routing_iter=3):
            '''
            Dynamic routing.
            '''
            for iter_idx in range(routing_iter):
                #The 3D softmax proposed softmaxes along 3 dimensions. Output channel, width, and height dimensions. This operation is equivalent to
                #permute the tensor temporarily such that the 3 desired dimensions are the the far right axes, reshape them into 1 single dimension and
                #perform the existing softmax function in that dimension. This means that even the feature maps in the capsules are individually contributing
                #to the feature maps in the next layer.
                temp = self.B.permute(0, 5, 3, 1, 2, 4).contiguous().view(batch_size, self.conv_channel_in, 1, self.height*self.width*self.conv_channel_out)

                k = func.softmax(temp, dim=-1) #apply softmax on the last dimension.

                #After the softmax, we can reshape and permute the tensor to the way it was.
                k = k.view(batch_size, self.conv_channel_in, 1, self.width, self.height, self.conv_channel_out).permute(0, 3, 4, 2, 5, 1).contiguous()

                S_tmp = k*x

                S = torch.sum(S_tmp, dim=-1, keepdim=True)

                S_hat = squash(S, dim=3) #squashing along the capsule's dimension.

                if iter_idx < routing_iter - 1:

                    agreements = (S_hat * x).sum(dim=3, keepdim=3)
                    self.B = self.B + agreements

            S_hat = S_hat.squeeze(-1)

            return S_hat.permute(0, 4, 3, 1, 2).contiguous()
















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
        self.conv3dcaps_31 = Conv3DCaps(height=2, width=2, conv_channel_in=32, caps_num_in=8, conv_channel_out=32, caps_num_out=8)
        self.conv2dcaps_32 = Conv2DCaps(height=2, width=2, conv_channel_in=32, caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=1)
        self.conv2dcaps_33 = Conv2DCaps(height=2, width=2, conv_channel_in=32,  caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=1)




        def flatten_caps(self, x):
            '''
            Removes spatial relationship between adjacent capsules while keeping the part-whole relationships between the capsules in the previous
            layer and the following layer before/after flatten_caps process.
            '''
            batch_size, _, dimensions, _, _ = x.size()
            x = x.permute(0, 3, 4, 1, 2).contiguous()
            return x.view(batch_size, -1, dimensions)







