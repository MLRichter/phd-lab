from torchvision.ops.misc import SqueezeExcitation, ConvNormActivation
from torch import nn


class Stem(nn.Module):

    def __init__(self):
        super(Stem, self).__init__()
        self.conv1 = ConvNormActivation(3, 256, kernel_size=3, stride=1,  bias=False)

    def forward(self, x):
        return self.conv1(x)


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, scale, final_layer=True):
        super(Block, self).__init__()
        self.conv1 = ConvNormActivation(in_channels, out_channels, kernel_size=scale, stride=scale, bias=False)
        self.conv2 = ConvNormActivation(out_channels, out_channels, kernel_size=scale, stride=1, bias=False)
        self.final_layer = final_layer
        if final_layer:
            self.conv3 = ConvNormActivation(out_channels, out_channels, kernel_size=scale, stride=1, bias=False)

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=scale, bias=False),
        )

    def forward(self, x):
        skip = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        if self.final_layer:
            x = self.conv3(x)
        x = x + skip
        return x


class FatNet(nn.Module):

    def __init__(self):
        super(FatNet, self).__init__()
        self.stem = Stem()
        self.block0 = Block(256, 512, 5)
        self.block1 = Block(512, 1024, 3)
        self.block2 = Block(1024, 1024, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

def fatnet(**kwargs):
    model = FatNet()
    model.name = "FatNet"
    return model


if __name__ == '__main__':
    from rfa_toolbox import create_graph_from_pytorch_model, visualize_architecture
    net = FatNet()
    g = create_graph_from_pytorch_model(net)
    visualize_architecture(g, "name").view()