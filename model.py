import torch
import torch.nn as nn
from clip import clip
import torchvision.models as models

class BottleneckCLIP(nn.Module):
    def __init__(self, output_dim, bias=False):
        super(BottleneckCLIP, self).__init__()
        model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
        in_features = 512  # Change to match ViT-B/32
        out_features = 1024

        # Take the visual encoder of CLIP and convert it to 32 bit
        self.encoder = model.visual.float()

        # Add a bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(in_features, in_features // 2, bias=bias),
            nn.BatchNorm1d(in_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features // 2, in_features // 2, bias=bias),
            nn.BatchNorm1d(in_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features // 2, out_features, bias=bias),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(out_features, output_dim)
        )

        # Assign preprocess object to self
        self.preprocess = preprocess


    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        x = self.bottleneck(x)
        return x

def get_CLIP_model(output_dim):
    model = BottleneckCLIP(output_dim)
    # Freeze CLIP weights
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("CLIP model ready to go!")
    return model.to("cuda" if torch.cuda.is_available() else "cpu"), model.preprocess


def get_Vgg19_model(output_dim):
    model = models.vgg19(pretrained=True)

    # Freeze training for all "features" layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Modify the classifier
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, output_dim)

    print("Vgg19 model ready to go!")
    return model.to("cuda" if torch.cuda.is_available() else "cpu")


def get_ResNet50_model(output_dim=102, dropout_p=0.5):
    # Load the pretrained ResNet50 model
    model = models.resnet50(pretrained=True)

    # Freeze training for all layers
    for param in model.parameters():
        param.requires_grad = False

    # Define a new head for the ResNet50 model
    class CustomResNet50(nn.Module):
        def __init__(self, base_model, output_dim, dropout_p):
            super(CustomResNet50, self).__init__()
            self.base_model = base_model
            self.base_model.fc = nn.Identity()  # Remove the original fully connected layer
            self.additional_layers = nn.Sequential(
                nn.Conv2d(2048, 1024, kernel_size=1),  # Additional conv layer
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),  # Ensure the same output size
                nn.Flatten(),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
                nn.Linear(512, output_dim)
            )

        def forward(self, x):
            x = self.base_model(x)
            x = x.view(x.size(0), 2048, 1, 1)  # Reshape the tensor to [batch_size, 2048, 1, 1]
            x = self.additional_layers(x)
            return x

    # Create an instance of the custom model
    custom_model = CustomResNet50(model, output_dim, dropout_p)

    print("Custom ResNet50 model ready to go!")
    return custom_model.to("cuda" if torch.cuda.is_available() else "cpu")