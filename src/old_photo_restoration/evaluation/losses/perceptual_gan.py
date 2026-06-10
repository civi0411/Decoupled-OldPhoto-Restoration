import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ==========================================
# 1. HỖ TRỢ TRÍCH XUẤT ĐẶC TRƯNG (VGG-19)
# ==========================================
class VGG19FeatureExtractor(nn.Module):
    """
    Dùng để tính Perceptual Loss theo thiết kế của Module 2.
    Sử dụng các tầng relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 của VGG19.
    """
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        # Tải trọng số VGG19 pre-trained
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # Chia model thành các block để lấy feature tại nhiều mức độ
        self.slices = nn.ModuleList([
            vgg19[:2],   # relu1_1
            vgg19[2:7],  # relu2_1
            vgg19[7:12], # relu3_1
            vgg19[12:21],# relu4_1
            vgg19[21:30] # relu5_1
        ])
        
        # Đóng băng toàn bộ trọng số (Chỉ dùng VGG để chấm điểm, không train VGG)
        for param in self.parameters():
            param.requires_grad = False
            
        # Thông số chuẩn hóa của ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # x được giả định có dải giá trị [0, 1]
        x = (x - self.mean) / self.std
        features = []
        for slice_layer in self.slices:
            x = slice_layer(x)
            features.append(x)
        return features


# ==========================================
# 2. HÀM LOSS CHO GENERATOR (Mạng LaMa chính)
# ==========================================
class LaMaGeneratorLoss(nn.Module):
    """
    Hàm Loss tổng hợp cho mạng LaMa Generator (Trục kép Twin-Bus).
    Bao gồm: L1 (Pixel) + Perceptual (Texture) + Adversarial (Độ chân thực).
    """
    def __init__(self, l1_weight: float = 10.0, perc_weight: float = 10.0, adv_weight: float = 1.0):
        super(LaMaGeneratorLoss, self).__init__()
        self.l1_weight = l1_weight
        self.perc_weight = perc_weight
        self.adv_weight = adv_weight
        
        self.l1_loss = nn.L1Loss()
        self.vgg_extractor = VGG19FeatureExtractor()
        # PatchGAN thường dùng BCE hoặc MSE cho Adversarial loss. Ở đây dùng BCE_With_Logits.
        self.adv_loss = nn.BCEWithLogitsLoss()

    def forward(self, img_restored: torch.Tensor, img_gt: torch.Tensor, disc_preds_fake: torch.Tensor):
        """
        img_restored: Ảnh do LaMa sinh ra (đã lấp nứt).
        img_gt: Ảnh sạch gốc (Ground Truth).
        disc_preds_fake: Nhận xét của PatchGAN về ảnh img_restored (dạng grid logit).
        """
        # 1. L1 Loss: Đảm bảo sai lệch pixel là nhỏ nhất
        l1 = self.l1_loss(img_restored, img_gt)
        
        # 2. Perceptual Loss: So sánh Feature Map (Trục kép Twin-Bus)
        feat_restored = self.vgg_extractor(img_restored)
        feat_gt = self.vgg_extractor(img_gt)
        
        perceptual = 0.0
        for f_res, f_gt in zip(feat_restored, feat_gt):
            perceptual += self.l1_loss(f_res, f_gt) # L1 trên không gian feature
            
        # 3. Adversarial Loss (Generator cố lừa Discriminator rằng ảnh sinh ra là Thật - nhãn 1)
        # Vì dùng PatchGAN, mục tiêu không phải là 1 con số, mà là 1 ma trận toàn số 1
        target_real = torch.ones_like(disc_preds_fake)
        adv = self.adv_loss(disc_preds_fake, target_real)
        
        # 4. Tổng hợp
        total_loss = (self.l1_weight * l1) + (self.perc_weight * perceptual) + (self.adv_weight * adv)
        
        loss_dict = {
            'gen_total': total_loss.item(),
            'l1': l1.item(),
            'perceptual': perceptual.item(),
            'adv_g': adv.item()
        }
        return total_loss, loss_dict


# ==========================================
# 3. HÀM LOSS CHO DISCRIMINATOR (PatchGAN)
# ==========================================
class PatchGANDiscriminatorLoss(nn.Module):
    """
    Hàm Loss để huấn luyện mạng PatchGAN Discriminator.
    Nhiệm vụ: Chấm đúng ảnh GT là Thật (1) và ảnh Restored là Giả (0).
    """
    def __init__(self):
        super(PatchGANDiscriminatorLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, disc_preds_real: torch.Tensor, disc_preds_fake: torch.Tensor):
        """
        disc_preds_real: Nhận xét của PatchGAN về ảnh Ground Truth.
        disc_preds_fake: Nhận xét của PatchGAN về ảnh Restored (đã bóc tách gradient bằng .detach()).
        """
        # Discriminator muốn nhận diện ảnh Real là 1
        target_real = torch.ones_like(disc_preds_real)
        loss_real = self.loss_fn(disc_preds_real, target_real)
        
        # Discriminator muốn nhận diện ảnh Fake là 0
        target_fake = torch.zeros_like(disc_preds_fake)
        loss_fake = self.loss_fn(disc_preds_fake, target_fake)
        
        # Trung bình cộng
        total_loss = (loss_real + loss_fake) * 0.5
        
        loss_dict = {
            'disc_total': total_loss.item(),
            'd_real': loss_real.item(),
            'd_fake': loss_fake.item()
        }
        return total_loss, loss_dict