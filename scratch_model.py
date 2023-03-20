import torch.nn as nn
import torch

nz = 100
ngf = 64
nc = 1
ndf = 8

noise = torch.Tensor(torch.ones(32, 100)).cuda()
g_layer1 = nn.Linear(nz, 7*7*ngf*8).cuda()
g_layer2 = nn.ConvTranspose2d(
    in_channels=ngf*8, out_channels=ngf*4, 
    kernel_size=3, stride=2, padding=1, output_padding=1
    ).cuda()
g_layer3 = nn.ConvTranspose2d(
    in_channels=ngf*4, out_channels=ngf*2, 
    kernel_size=3, stride=1, padding=1
    ).cuda()
g_layer4 = nn.ConvTranspose2d(
    in_channels=ngf*2, out_channels=ngf, 
    kernel_size=3, stride=1, padding=1
    ).cuda()
g_layer5 = nn.ConvTranspose2d(
    in_channels=ngf, out_channels=nc, 
    kernel_size=3, stride=2, padding=1, output_padding=1
    ).cuda()
g_layer1_result = g_layer1(noise)  ## g_layer1_result.shape = torch.Size([32, 25088])
g_layer1_result = g_layer1_result.view(g_layer1_result.size()[0], 512, 7, 7)
g_layer2_result = g_layer2(g_layer1_result)
g_layer3_result = g_layer3(g_layer2_result)
g_layer4_result = g_layer4(g_layer3_result)
g_layer5_result = g_layer5(g_layer4_result)
print("noise: ", noise.shape)
print("g_layer1_result: ", g_layer1_result.shape)
print("g_layer2_result: ", g_layer2_result.shape)
print("g_layer3_result: ", g_layer3_result.shape)
print("g_layer4_result: ", g_layer4_result.shape)
print("g_layer5_result: ", g_layer5_result.shape)


sample = torch.Tensor(torch.ones(32, 1, 28, 28)).cuda()
d_layer1 = nn.Conv2d(
    in_channels=nc, out_channels=ndf, 
    kernel_size=3, padding=1
    ).cuda()
d_layer2 = nn.Conv2d(
    in_channels=ndf, out_channels=ndf*2, 
    kernel_size=3, stride=2, padding=1
    ).cuda()
d_layer3 = nn.Conv2d(
    in_channels=ndf*2, out_channels=ndf*4, 
    kernel_size=3, stride=2,padding=1
    ).cuda()
d_layer4 = nn.Conv2d(
    in_channels=ndf*4, out_channels=ndf*8, 
    kernel_size=3, padding=1
    ).cuda()
print("check")
dis_layer = nn.Sequential(nn.Linear(in_features=64*7*7, out_features=1).cuda(), nn.Sigmoid())
d_layer1_result = d_layer1(sample)
d_layer2_result = d_layer2(d_layer1_result)
d_layer3_result = d_layer3(d_layer2_result)
d_layer4_result = d_layer4(d_layer3_result)
d_layer4_result_mod = d_layer4_result.view(d_layer4_result.size()[0], -1)
d_dis_layer_result= dis_layer(d_layer4_result_mod)

print("sample: ", sample.shape)
print("d_layer1_result: ", d_layer1_result.shape)
print("d_layer2_result: ", d_layer2_result.shape)
print("d_layer3_result: ", d_layer3_result.shape)
print("d_layer4_result: ", d_layer4_result.shape)
print("d_dis_layer_result: ", d_dis_layer_result.shape)