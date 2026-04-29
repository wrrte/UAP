import numpy as np
import torch
import torch.nn as nn
import scipy.stats as st
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F

####################################################################################################
# [ALL] preparing the current input tensor for gradient computation
def prepare_attack_input(x_adv):
    x_adv = x_adv.detach()
    x_adv.requires_grad = True
    return x_adv

# [ALL] generating the base gradient(ghat) for the current step
def calculate_attack_ghat(model, x_adv_or_nes, y, number_of_si_scales, target_label, 
                          attack_type, di_prob, di_pad_amount, di_pad_value, feature_attack, x_clean, depth):
    
    si_ghat = apply_si(model, x_adv_or_nes, y, number_of_si_scales, target_label, 
                       attack_type, di_prob, di_pad_amount, di_pad_value, feature_attack, x_clean, depth)
    if si_ghat is not None:
        return si_ghat

    attack_input = apply_di(x_adv_or_nes, attack_type, di_prob, di_pad_amount, di_pad_value)
    return calculate_loss_gradient(model, attack_input, x_adv_or_nes, y, target_label, feature_attack, x_clean, depth)

def extract_features(model, input, depth):
    features = []
    def hook(module, input, output):
        features.append(output)
    module = model
    for key in depth.split('.'):
        module = module._modules[key]
    handle = module.register_forward_hook(hook)
    model(input)
    handle.remove()
    return features

# [TODO 2] Feature attack loss
def calculate_loss_gradient(model, model_input, grad_input, y, target_label, feature_attack, clean_input, depth):
    output = model(model_input)
    if feature_attack==False:
        loss = nn.CrossEntropyLoss()(output, y)
    if feature_attack==True:
        clean_features = extract_features(model, clean_input, depth)[0]
        adv_features = extract_features(model, model_input, depth)[0]
        
        clean_features = clean_features.view(clean_features.size(0), -1)
        adv_features = adv_features.view(adv_features.size(0), -1)
        
        loss = -F.cosine_similarity(adv_features, clean_features).mean()
        
    if target_label >= 0:
        loss = -loss
    return torch.autograd.grad(loss, grad_input, retain_graph=False, create_graph=False)[0]
####################################################################################################



####################################################################################################
# [MI] configures MI only when M is provided, and sets decay rate via the mu parameter
def apply_mi(attack_type, mu):
    if "M" not in attack_type:
        return 0
    return mu

def update_mi_momentum(g, ghat, mu):
    norm = torch.sum(torch.abs(ghat), dim=(1, 2, 3), keepdim=True)
    norm = torch.clamp(norm, min=1e-12)
    normalized_ghat = ghat / norm
    g = mu * g + normalized_ghat
    return g
####################################################################################################



####################################################################################################
# [DI] configures DI only when D is provided
def apply_di(x_adv, attack_type, di_prob, di_pad_amount, di_pad_value):
    if 'D' in attack_type:
        return diverse_input(x_adv, di_prob, di_pad_amount, di_pad_value)
    return x_adv

# [DI] Implementing diverse input (resize & padding)
def diverse_input(x_adv, di_prob, di_pad_amount, di_pad_value):
    if torch.rand(1).item() < di_prob:
        _, _, h, w = x_adv.size()
        rnd = torch.randint(h, h + di_pad_amount + 1, (1,)).item()
        rescaled = F.interpolate(x_adv, size=(rnd, rnd), mode='bilinear', align_corners=False)
        
        pad_top = torch.randint(0, h + di_pad_amount - rnd + 1, (1,)).item()
        pad_bottom = h + di_pad_amount - rnd - pad_top
        pad_left = torch.randint(0, w + di_pad_amount - rnd + 1, (1,)).item()
        pad_right = w + di_pad_amount - rnd - pad_left
        
        padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom), value=di_pad_value)
        x_di = F.interpolate(padded, size=(h, w), mode='bilinear', align_corners=False)
        return x_di
    return x_adv
####################################################################################################



####################################################################################################
# [TI] configures TI only when T is provided 
# (FYI. ti_conv is a smoothed gradient generated via the create_ti_conv function.)
def apply_ti(ghat, attack_type, ti_conv):
    if 'T' in attack_type:
        return ti_conv(ghat)
    return ghat

# [TI] creating Gaussian kernel
def gkern(kernlen=7, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel.astype(np.float32)

# [TI] preparing depthwise convolution
def create_ti_conv(device, ti_kernel_size):
    kernel = gkern(ti_kernel_size, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)
    ti_conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(ti_kernel_size, ti_kernel_size),
                              padding=ti_kernel_size // 2, groups=3, bias=False)
    with torch.no_grad():
        ti_conv.weight = nn.Parameter(torch.from_numpy(stack_kernel).float().to(device))
        ti_conv.requires_grad_(False)
    return ti_conv.to(device)
####################################################################################################



####################################################################################################
# [SI] configures SI only when S is provided
def apply_si(model, x_adv_or_nes, y, number_of_si_scales, target_label, attack_type, di_prob, di_pad_amount,
             di_pad_value, feature_attack, x_clean, depth):
    if 'S' in attack_type:
        return calculate_si_ghat(model, x_adv_or_nes, y, number_of_si_scales, target_label, attack_type, di_prob,
                                 di_pad_amount, di_pad_value, feature_attack, x_clean, depth)
    return None

# [SI] accumulates gradients across multi-scale inputs (SI), with optional Diverse Input (DI) support via apply_di
def calculate_si_ghat(model, x_adv_or_nes, y, number_of_si_scales, target_label, 
                      attack_type, di_prob, di_pad_amount, di_pad_value, feature_attack, x_clean, depth):
    grad_sum = 0
    for si_counter in range(number_of_si_scales):
        si_div = 2 ** si_counter
        scaled_x = x_adv_or_nes / si_div
        
        scaled_x = scaled_x.detach()
        scaled_x.requires_grad = True
        
        di_x = apply_di(scaled_x, attack_type, di_prob, di_pad_amount, di_pad_value)
        grad = calculate_loss_gradient(model, di_x, scaled_x, y, target_label, feature_attack, x_clean, depth)
        
        # 잃어버렸던 핵심 가중치 연산 복구
        grad_sum = grad_sum + grad * (1.0 / si_div)
        
    return grad_sum
####################################################################################################

####################################################################################################
# [NI] configures NI only when N is provided, and prepares the look-ahead input tensor
def apply_ni(attack_type, x_adv, alpha, mu, g):
    if "N" in attack_type:
        x_nes = x_adv + alpha * mu * g
        # 잃어버렸던 핵심 Gradient 활성화 함수 복구
        return prepare_attack_input(x_nes)
    return prepare_attack_input(x_adv)

def apply_ni_decay(attack_type, mu):
    if "N" not in attack_type:
        return 0
    return mu
####################################################################################################
