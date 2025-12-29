import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# PREPROCESSING — SAME FOR ALL MODELS (RESIZE → 512)
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==================================================================
# ==========================  CBAM MODULES  =========================
# ==================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, 1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


class CBAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.channel_att = ChannelAttention(in_planes)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        out = x * self.channel_att(x)
        out = out * self.spatial_att(out)
        return out


# ==================================================================
# ==========  MODEL DEFINITIONS (3 Different Architectures)  ========
# ==================================================================

# -------------------------
# 1. EfficientNetB0 + CBAM → For Bladder
# -------------------------
class EfficientNetB0_CBAM(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        self.features = self.backbone.features
        self.cbam = CBAM(1280)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return self.classifier(x)


# -------------------------
# 2. MobileNetV2 → For Ureter
# -------------------------
def MobileNetV2_Model(num_classes=2):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(1280, num_classes)
    return model


# -------------------------
# 3. ResNet18 → For Kidney
# -------------------------
def ResNet18_Model(num_classes=2):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, num_classes)
    return model


# ==================================================================
# ======================  LOAD ALL MODELS HERE  =====================
# ==================================================================

# --- Replace with your paths ---
bladder_model_path = r"E:\Yolo train\Aug Bladder Crop\best_efficientnetb0_binary_BladderCrop_CBAM aug.pth"
ureter_model_path  = r"E:\Yolo train\Aug ureter Crop\best_mobilenetv2_binary ureter Crop_1 aug.pth"
kidney_model_path  = r"E:\Yolo train\AUg Kidney Crop\best_resnet18_binary_kidney_crop aug.pth"

# Load Bladder Model (EffNetB0 + CBAM)
bladder_model = EfficientNetB0_CBAM(pretrained=False).to(DEVICE)
bladder_model.load_state_dict(torch.load(bladder_model_path, map_location=DEVICE))
bladder_model.eval()

# Load Ureter Model (MobileNetV2)
ureter_model = MobileNetV2_Model().to(DEVICE)
ureter_model.load_state_dict(torch.load(ureter_model_path, map_location=DEVICE))
ureter_model.eval()

# Load Kidney Model (ResNet18)
kidney_model = ResNet18_Model().to(DEVICE)
kidney_model.load_state_dict(torch.load(kidney_model_path, map_location=DEVICE))
kidney_model.eval()

def generate_xai(model, img_tensor, orig_img, save_path):
    # choose last conv layer
    if hasattr(model, "features"):
        target_layers = [model.features[-1]]
    else:
        target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(
        input_tensor=img_tensor,
        targets=[ClassifierOutputTarget(1)]
    )[0]

    orig = np.array(orig_img.resize((512, 512))) / 255.0
    heatmap = show_cam_on_image(orig, grayscale_cam, use_rgb=True)

    xai_file = save_path.replace(".jpg", "_xai.jpg")
    Image.fromarray(heatmap).save(xai_file)

    return xai_file

# ==================================================================
# =========================  FINAL PREDICT  =========================
# ==================================================================
def classify_image(image_path, organ_name, generate_xai_flag=False):
    img = Image.open(image_path).convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    organ = organ_name.lower()

    # select model
    if organ == "bladder":
        model = bladder_model
    elif organ == "ureter":
        model = ureter_model
    elif organ == "kidney":
        model = kidney_model
    else:
        return "Unknown Organ", None

    # run prediction
    output = model(img_tensor)
    _, pred = torch.max(output, 1)
    label = "Stone" if pred.item() == 1 else "Normal"

    # return normal prediction only
    if not generate_xai_flag:
        return label

    # generate XAI
    xai_image_path = image_path.replace(".jpg", "_xai.jpg")
    xai_image_path = generate_xai(model, img_tensor, img, xai_image_path)

    return label, xai_image_path


    _, pred = torch.max(output, 1)
    return "Stone" if pred.item() == 1 else "Normal"
