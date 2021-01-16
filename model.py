'''
PyTorch implementation of Deep-CapsNet architecture.
'''

import math
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
        super(ConvertToCaps, self).__init__()

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

    def __init__(self, height, width, conv_channel_in, caps_num_in, conv_channel_out, caps_num_out, device, kernel_size=3, stride=1, routing_iter=1, pad=1):
        '''
        Parameter Init.
        '''

        super(Conv2DCaps, self).__init__()
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
        self.device = device

        #Capsule 2D convolution works by temporarily reshaping the capsule tensors[batch size, channels, num capsules, feature width, feature height]
        #back to conv tensors [batch size, channels, feature width, feature height] and perform a typical 2D Convolution process. The output of this
        #process is then converted back to capsule tensors. The final capsule tensors will undergo the squash activation function before being returned.

        #The given capsule tensor inputs [batch size, channels, num capsules, feature width, feature height] is reshaped into conv tensors by
        #treating the channels*numcapsules as the new channel input. The output of the conv operation would be also treated as such which enables
        #the final output channels to be reshaped back into the desired num of capsules.
        reshaped_in_channels = self.conv_channel_in*self.caps_num_in
        reshaped_out_channels= self.conv_channel_out*self.caps_num_out

        self.conv = nn.Conv2d(in_channels=reshaped_in_channels, out_channels=reshaped_out_channels, kernel_size=self.kernel_size, stride=self.stride,
                            padding=self.pad).to(self.device)


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

    def __init__(self, height, width, conv_channel_in, caps_num_in, conv_channel_out, caps_num_out, device, kernel_size=3, routing_iter=3):
        '''
        Parameter Init.
        '''
        super(Conv3DCaps, self).__init__()

        self.height = height
        self.width = width
        self.conv_channel_in = conv_channel_in
        self.caps_num_in = caps_num_in
        self.conv_channel_out = conv_channel_out
        self.caps_num_out = caps_num_out
        self.kernel_size = kernel_size
        self.routing_iter = routing_iter
        self.device = device

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

        self.conv_3d = nn.Conv3d(in_channels=reshaped_in_channels,
                                out_channels=reshaped_out_channels,
                                kernel_size=self.kernel_size,
                                padding=pad, stride=stride).to(self.device)


    def forward(self, x):
        '''
        Forward Propagation.
        '''
        batch_size = x.size()[0]
        x = x.view(batch_size, self.conv_channel_in*self.caps_num_in, self.height, self.width)
        x = x.unsqueeze(1)
        x = self.conv_3d(x)
        self.height, self.width = x.size()[-2:]

        #ALL the permute operations are done according to the paper.
        x = x.permute(0, 2, 1, 3, 4)
        x = x.view(batch_size, self.conv_channel_in, self.conv_channel_out, self.caps_num_out, self.height, self.width)

        x = x.permute(0, 4, 5, 3, 2, 1).contiguous()
        x_detached = x.detach()
        self.B = x_detached.new_zeros(size=(batch_size, self.width, self.height, 1, self.conv_channel_out,
                                self.conv_channel_in), requires_grad=False).to(self.device)

        x = self.routing(x, x_detached, batch_size, self.routing_iter)

        return x



    def routing(self, x, x_detached, batch_size, routing_iter=3):
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

            if iter_idx == routing_iter-1:
                S = (k*x).sum(dim=-1, keepdim=True)
                S_hat = squash(S.permute(0, 4, 3, 1, 2, 5).contiguous(), dim=2)
                # S_hat = squash(S, dim=-1)

            else:
                S = (k*x_detached).sum(dim=-1, keepdim=True)
                tmp_S = squash(S.permute(0, 4, 3, 1, 2, 5).contiguous(), dim=2)
                S_hat = tmp_S.permute(0, 3, 4, 2, 1, 5).contiguous()
                # S_hat = squash(S, dim=-1)
                agreements = (S_hat * x_detached).sum(dim=3, keepdim=True)
                self.B = self.B + agreements


        S_hat = S_hat.squeeze(-1)

        # return S_hat.permute(0, 4, 3, 1, 2).contiguous()
        return S_hat



class FC_Caps(nn.Module):

    def __init__(self, output_capsules, input_capsules, in_dimensions, out_dimensions, device, routing_iter=3):
        '''
        Param init.
        '''

        super(FC_Caps, self).__init__()

        self.output_capsules = output_capsules
        self.input_capsules = input_capsules
        self.in_dimensions = in_dimensions
        self.out_dimensions = out_dimensions
        self.routing_iter = routing_iter
        self.device = device

        self.W = nn.Parameter(torch.randn(1, self.input_capsules, self.output_capsules, self.out_dimensions, self.in_dimensions)*0.05)
        self.b = nn.Parameter(torch.randn(1, 1, self.output_capsules, self.out_dimensions)*0.05)


    def forward(self, x):
        '''
        Forward propagation with dynamic routing as proposed in the original 2017 paper.
        '''

        x = x.unsqueeze(dim=2).unsqueeze(dim=4)

        u_hat = torch.matmul(self.W, x).squeeze()

        u_hat_detached = u_hat.detach()

        b_ij = x.new_zeros(size=(x.size()[0], self.input_capsules, self.output_capsules, 1), requires_grad=False).to(self.device)

        #Dynamic routing
        for iter_idx in range(self.routing_iter):

            c_ij = func.softmax(b_ij, dim=2)

            if iter_idx == self.routing_iter - 1:
                s_j = (c_ij * u_hat).sum(dim=1, keepdim=True) + self.b #multiply with the original u_hat since we want the gradient flow.
                v_j = squash(s_j, dim=-1)

            else:
                s_j = (c_ij*u_hat_detached).sum(dim=1, keepdim=True) #no gradient flow.
                v_j = squash(s_j, dim=-1)
                a_ij = (u_hat_detached * v_j).sum(dim=-1, keepdim=True) #agreement check
                b_ij = b_ij + a_ij #update the coefficients

        return v_j.squeeze()


class Mask_CID(nn.Module):
    '''
    Masks out all capsules except the capsules that represent the class.
    '''

    def __init__(self, device):

        super(Mask_CID, self).__init__()
        self.device = device

    def forward(self, x, target=None):

        batch_size = x.size()[0]

        classes = torch.norm(x, dim=2)
        if target is None:
            max_len_indices = classes.max(dim=1)[1].squeeze()
        else:
            max_len_indices = target.max(dim=1)[1]

        batch_ind = torch.arange(start=0, end=batch_size).to(self.device) #a tensor containing integer from 0 to batch size.
        m = torch.stack([batch_ind, max_len_indices], dim=1).to(self.device) #records the label's index for every batch.
        masked = torch.zeros((batch_size, 1) + x.size()[2:]).to(self.device)

        for i in range(batch_size):
            masked[i] = x[m[i][0], m[i][1], :].unsqueeze(0)

        if target is None:
            return masked.squeeze(-1), max_len_indices
        return masked.squeeze(-1), classes.max(dim=1)[1].squeeze()



class Decoder(nn.Module):
    '''
    Reconstruct back the input image from the prediction capsule using transposed Convolutions.
    '''

    def __init__(self, caps_dimension, device, num_caps=1, img_size=28, img_channels=1):

        super(Decoder, self).__init__()

        self.num_caps = num_caps
        self.img_channels = img_channels
        self.img_size = img_size
        self.caps_dimension = caps_dimension
        self.neurons = self.img_size//4
        self.device = device

        self.fc = nn.Sequential(torch.nn.Linear(self.caps_dimension*self.num_caps, self.neurons*self.neurons*16), nn.ReLU(inplace=True)).to(self.device)

        self.reconst_layers1 = nn.Sequential(nn.BatchNorm2d(num_features=16, momentum=0.8),
                                                nn.ConvTranspose2d(in_channels=16, out_channels=64,
                                                kernel_size=3, stride=1, padding=1))

        self.reconst_layers2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.reconst_layers3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.reconst_layers4 = nn.Sequential(nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
                                                nn.ReLU(inplace=True))


    def forward(self, x):
        '''
        Forward Propagation
        '''

        x = x.type(torch.FloatTensor).to(self.device)

        x = self.fc(x)
        x = x.reshape(-1, 16, self.neurons, self.neurons)
        x = self.reconst_layers1(x)
        x = self.reconst_layers2(x)

        p2d = (1, 0, 1, 0)
        x = func.pad(x, p2d, "constant", 0)
        x = self.reconst_layers3(x)

        x = func.pad(x, p2d, "constant", 0)
        x = self.reconst_layers4(x)

        x = x.view(-1, 1, self.img_size, self.img_size)

        return x


class DeepCapsModel(nn.Module):
    '''
    DeepCaps Model.
    '''

    def __init__(self, num_class, img_height, img_width, device):
        '''
        Init the architecture and parameters.
        '''

        super(DeepCapsModel, self).__init__()

        self.num_class = num_class
        self.height, self.width = img_height, img_width

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(num_features=128, eps=1e-08, momentum=0.99)

        self.toCaps = ConvertToCaps()

        self.conv2dcaps_00 = Conv2DCaps(height=self.height, width=self.width, conv_channel_in=128, caps_num_in=1, conv_channel_out=32,
                                        caps_num_out=4, stride=2, device=device)
        height, width = math.ceil(self.height/2), math.ceil(self.width/2)
        self.conv2dcaps_01 = Conv2DCaps(height=height, width=width, conv_channel_in=32, caps_num_in=4, conv_channel_out=32,
                                        caps_num_out=4, stride=1, device=device)
        self.conv2dcaps_02 = Conv2DCaps(height=height, width=width, conv_channel_in=32, caps_num_in=4, conv_channel_out=32,
                                        caps_num_out=4, stride=1, device=device)
        self.conv2dcaps_03 = Conv2DCaps(height=height, width=width, conv_channel_in=32, caps_num_in=4, conv_channel_out=32,
                                        caps_num_out=4, stride=1, device=device)


        self.conv2dcaps_10 = Conv2DCaps(height=height, width=width, conv_channel_in=32, caps_num_in=4, conv_channel_out=32,
                                        caps_num_out=8, stride=2, device=device)
        height, width = math.ceil(height/2), math.ceil(width/2)
        self.conv2dcaps_11 = Conv2DCaps(height=height, width=width, conv_channel_in=32, caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=1, device=device)
        self.conv2dcaps_12 = Conv2DCaps(height=height, width=width, conv_channel_in=32, caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=1, device=device)
        self.conv2dcaps_13 = Conv2DCaps(height=height, width=width, conv_channel_in=32, caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=1, device=device)
        self.conv2dcaps_13 = Conv2DCaps(height=height, width=width, conv_channel_in=32, caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=1, device=device)


        self.conv2dcaps_20 = Conv2DCaps(height=height, width=width, conv_channel_in=32, caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=2, device=device)
        height, width = math.ceil(height/2), math.ceil(width/2)
        self.conv2dcaps_21 = Conv2DCaps(height=height, width=width, conv_channel_in=32, caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=1, device=device)
        self.conv2dcaps_22 = Conv2DCaps(height=height, width=width, conv_channel_in=32, caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=1, device=device)
        self.conv2dcaps_23 = Conv2DCaps(height=height, width=width, conv_channel_in=32, caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=1, device=device)


        self.conv2dcaps_30 = Conv2DCaps(height=height, width=width, conv_channel_in=32, caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=2, device=device)
        height, width = math.ceil(height/2), math.ceil(width/2)
        self.conv3dcaps_31 = Conv3DCaps(height=height, width=width, conv_channel_in=32, caps_num_in=8, conv_channel_out=32, caps_num_out=8, device=device)
        self.conv2dcaps_32 = Conv2DCaps(height=height, width=width, conv_channel_in=32, caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=1, device=device)
        self.conv2dcaps_33 = Conv2DCaps(height=height, width=width, conv_channel_in=32,  caps_num_in=8, conv_channel_out=32,
                                        caps_num_out=8, stride=1, device=device)

        self.fc_caps = FC_Caps(output_capsules=self.num_class, input_capsules=640, in_dimensions=8,
                                    out_dimensions=16, routing_iter=3, device=device)

        self.mask = Mask_CID(device=device)
        self.decoder = Decoder(caps_dimension=16, num_caps=1, device=device, img_size=self.width, img_channels=1)
        self.mse_loss = nn.MSELoss(reduction='none')


    def forward(self, x, target=None):
        '''
        Forward Propagation of DeepCaps Model.
        '''

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.toCaps(x)

        x = self.conv2dcaps_00(x)
        x_skip = self.conv2dcaps_01(x)
        x = self.conv2dcaps_02(x)
        x = self.conv2dcaps_03(x)

        x = x + x_skip

        x = self.conv2dcaps_10(x)
        x_skip = self.conv2dcaps_11(x)
        x = self.conv2dcaps_12(x)
        x = self.conv2dcaps_13(x)

        x = x + x_skip

        x = self.conv2dcaps_20(x)
        x_skip = self.conv2dcaps_21(x)
        x = self.conv2dcaps_22(x)
        x = self.conv2dcaps_23(x)

        x = x + x_skip
        x1 = x

        x = self.conv2dcaps_30(x)
        x_skip = self.conv3dcaps_31(x)
        x = self.conv2dcaps_32(x)
        x = self.conv2dcaps_33(x)

        x = x + x_skip
        x2 = x

        xa = self.flatten_caps(x1)
        xb = self.flatten_caps(x2)

        x = torch.cat((xa, xb), dim=-2)
        dig_caps = self.fc_caps(x)

        x = self.to_scalar(dig_caps)

        masked, indices = self.mask(dig_caps, target)
        decoded = self.decoder(masked)

        return dig_caps, masked, decoded, indices



    def flatten_caps(self, x):
        '''
        Removes spatial relationship between adjacent capsules while keeping the part-whole relationships between the capsules in the previous
        layer and the following layer before/after flatten_caps process.
        '''
        batch_size, _, dimensions, _, _ = x.size()
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        return x.view(batch_size, -1, dimensions)

    def to_scalar(self, x):
        '''
        Calculate and returns the length of each capsule.
        '''
        return torch.norm(x, dim=2)


    def margin_loss(self, x, labels, lambda_, m_plus, m_minus):
        '''
        Classification loss.
        '''
        batch_size = x.size()[0]

        v_c = torch.norm(x, dim=2, keepdim=True)

        #we're using ReLU functions here because ReLU selects 0 when the tensor given is below 0.
        max_l = func.relu(m_plus - v_c).view(batch_size, -1)**2
        max_r = func.relu(v_c - m_minus).view(batch_size, -1)**2

        classification_loss = (labels*max_l + lambda_*(1-labels)*max_r).sum(dim=1)
        return classification_loss

    def reconstruction_loss(self, reconstructed, data):
        '''
        Reconstruction loss.
        '''
        batch_size = reconstructed.size()[0]
        loss = self.mse_loss(reconstructed.view(batch_size, -1), data.view(batch_size, -1))
        return 0.4 * loss.sum(dim=1)


    def loss(self, x, reconstructed, data, labels, lambda_=0.5, m_plus=0.9, m_minus=0.1):
        '''
        Mean of total loss calculation. Both reconstruction loss and classification loss.
        '''
        total_loss = self.margin_loss(x, labels, lambda_, m_plus, m_minus) + self.reconstruction_loss(reconstructed, data)
        return total_loss.mean()
