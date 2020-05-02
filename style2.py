import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms,models
from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import cv2


device = ("cuda" if torch.cuda.is_available() else "cpu")

class SaveFeatures(nn.Module):
	features = None;
	def __init__(self, m):
		self.hook = m.register_forward_hook(self.hook_fn)
	def hook_fn(self, module, input, output):
		self.features = output
	def close(self):
		self.hook.remove()

def imcnvt(image):
    x = image.to("cpu").clone().detach().numpy().squeeze()
    x = x.transpose(1,2,0)
    x=np.clip(x,0,1);
    return x

def load_images(idx):

    transform = transforms.Compose([transforms.Resize(480),transforms.ToTensor(),])

    content = Image.open("outputs_1/{}_input.png".format(idx)).convert("RGB")
    content = transform(content).to(device)
    style = Image.open("outputs_1/{}_style.png".format(idx)).convert("RGB")
    style = transform(style).to(device)
    
    mask = Image.open("outputs_1/{}_mask_dilated.png".format(idx)).convert("RGB")
    mask = transform(mask).float().to(device)
    mask_sth = Image.open("outputs_1/{}_mask.png".format(idx)).convert("RGB")
    mask_sth = transform(mask_sth).float().to(device)

    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(imcnvt(content),label = "Content")
    ax2.imshow(imcnvt(style),label = "Style")       
    plt.show()

    return content,style,mask,mask_sth

def halve_size(mask):
    h,w = mask.shape
    return cv2.resize(mask,(w//2,h//2))

ConvolMask = nn.AvgPool2d(3, 1, 1)
def convol(mask, nb):
    x = Variable(torch.tensor(mask[None][None]))
    for i in range(nb): x = ConvolMask(x)
    return x.data.squeeze().clone().detach().numpy()

def get_mask_ftrs(mask):
    ftrs = []
    mask = halve_size(convol(mask,2))
    mask = halve_size(convol(mask,2))
    mask = convol(mask,1)
    ftrs.append(mask)
    mask = halve_size(convol(mask,3))
    mask = convol(mask,1)
    ftrs.append(mask)
    mask = halve_size(convol(mask,3))
    mask = convol(mask,1)
    ftrs.append(mask)
    return ftrs


def get_patches(x,ks=3,stride=1,padding=1):
    ch, n1, n2 = x.shape
    y = np.zeros((ch,n1+2*padding,n2+2*padding))
    y[:,padding:n1+padding,padding:n2+padding] = x
    start_idx = np.array([j + (n2+2*padding)*i for i in range(0,n1-ks+1+2*padding,stride) for j in range(0,n2-ks+1+2*padding,stride) ])
    grid = np.array([j + (n2+2*padding)*i + (n1+2*padding) * (n2+2*padding) * k for k in range(0,ch) for i in range(ks) for j in range(ks)])
    to_take = start_idx[:,None] + grid[None,:]
    return y.take(to_take)

def match_ftrs(inp_ftrs,sty_ftrs):
    res = []
    for l_inp,s_inp in zip(inp_ftrs,sty_ftrs):
        l_inp = torch.tensor(get_patches(l_inp[0].data.clone().detach().numpy())).to(device)
        s_inp = torch.tensor(get_patches(s_inp[0].data.clone().detach().numpy())).to(device)
        scals = torch.mm(l_inp,s_inp.t())
        norms_in = torch.sqrt((l_inp ** 2).sum(1))
        norms_st = torch.sqrt((s_inp ** 2).sum(1))
        cosine_sim = scals / (1e-15 + norms_in.unsqueeze(1) * norms_st.unsqueeze(0))
        _, idx_max = cosine_sim.max(1)
        res.append(idx_max.clone().detach().numpy())
    return res

def map_style(style_ftrs,map_ftrs):
    res = []
    for sf, mapf in zip(style_ftrs, map_ftrs):
        sf = sf.clone().detach().numpy().reshape(sf.size(1),-1)
        sf = sf[:,mapf]
        res.append(torch.tensor(sf))
    return res

def hist_loss(source,target):
    shape=source.shape
    s_flatten = source.contiguous().view(-1)
    t_flatten = target.contiguous().view(-1)
    n_bins=255
    max_value=torch.max(torch.max(s_flatten),torch.max(t_flatten))
    min_value=torch.min(torch.min(s_flatten),torch.min(t_flatten))
    hist_delta=(max_value-min_value)/n_bins
    hist_range=torch.arange(min_value.item(),max_value.item(),hist_delta.item())
    s_hist=torch.histc(s_flatten, bins=n_bins, min=min_value.item(), max=max_value.item())
    t_hist=torch.histc(t_flatten, bins=n_bins, min=min_value.item(), max=max_value.item())
    return F.mse_loss(s_hist.detach(),t_hist.detach())/s_hist.numel()

def histogram_loss(out_ftrs):
    loss = 0
    for of, sf, mf in zip(out_ftrs, style_features, mask_ftrs):
        to_pass = of * torch.tensor(mf[None,None], requires_grad=False)
        to_pass = to_pass.view(to_pass.size(1),-1)
        sf = sf * torch.tensor(mf, requires_grad=False)
        loss += hist_loss(to_pass,sf)
    return loss

def hist_mask(sf, mf):
    res = []
    mask = torch.Tensor(mf).contiguous()
    masked = sf * mask
    return torch.cat([torch.histc(masked[i][mask>=0.1], 255).unsqueeze(0) for i in range(masked.size(0))]).to(device)

if __name__ == '__main__':
    content,style,mask,mask_sth=load_images(6)
    
    model = getattr(models, 'vgg19')
    vgg = model(pretrained=True) 

    vgg=nn.Sequential(*list(vgg.features.children())[:43]).to(device)
    for p in vgg.parameters():
        p.requires_grad = False

    # print(vgg)

    idx_layers=[11,20,29]
    sfs = [SaveFeatures(vgg[idx]) for idx in idx_layers]

    vgg(Variable(content[None].to(device)))
    content_features = [sf.features.clone() for sf in sfs]
    print([i.shape for i in content_features])
    
    vgg(Variable(style[None].to(device)))
    style_features = [sf.features.clone() for sf in sfs]
    
    mask_ftrs = get_mask_ftrs(mask[0,:,:])
    
    map_ftrs = match_ftrs(content_features, style_features)
    sty_ftrs = map_style(style_features,map_ftrs)
    

    max_iter = 1000
    show_iter = 10

    target = content.clone().unsqueeze(0).requires_grad_(True).to(device)
    optimizer = torch.optim.LBFGS([target],lr=1)
  

    def content_loss(out_ftrs):
        msk_of = out_ftrs * torch.tensor(mask_ftrs[1][None,None])
        msk_if = content_features[1] * mask_ftrs[1][None,None]
        return F.mse_loss(msk_of,msk_if, size_average=False) / float(out_ftrs.size(1) * mask_ftrs[1].sum())

    def step():
        global n_iter
        optimizer.zero_grad()
        loss = stage2_loss(target)
        loss.backward()
        if n_iter%show_iter==0:
            print(f'Iteration: {n_iter}, loss: {loss}')
            if n_iter % 50 == 0:
                # global mask_sth
                outimg=imcnvt(target)
                mask_sth1=imcnvt(mask_sth)
                style1=imcnvt(style)
                outimg=outimg * mask_sth1 + style1 * (1 - mask_sth1)
                outimg=outimg.clip(0,1)
                plt.imshow(outimg,label="Epoch "+str(n_iter))
                plt.show()
                plt.imsave('D:/project/outputs2/'+str(n_iter)+'.png',outimg,format='png')
        n_iter += 1
        return loss

    def gram(x):
        return torch.mm(x, x.t())

    def gram_mse_loss(input, target): return F.mse_loss(gram(input), gram(target))

    def style_loss(out_ftrs):
        loss = 0
        for of, sf, mf in zip(out_ftrs, sty_ftrs, mask_ftrs):
            to_pass = of * torch.tensor(mf[None,None], requires_grad=False)
            to_pass = to_pass.view(to_pass.size(1),-1)
            sf = sf * torch.tensor(mf, requires_grad=False).view(1,-1)
            loss += gram_mse_loss(to_pass,sf)
        return loss / 3
    
    w_c, w_s, w_h = 1, 10,1
    
    def stage2_loss(opt_img_v):
        vgg(opt_img_v)
        out_ftrs = [o.features for o in sfs]
        c_loss = content_loss(out_ftrs[1])
        s_loss = style_loss(out_ftrs)
        h_loss=histogram_loss(out_ftrs)
        return w_c * c_loss + w_s * s_loss + w_h * h_loss

    n_iter=0
    while n_iter <= max_iter: optimizer.step(step)