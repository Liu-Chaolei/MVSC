from torch.nn import functional as F
import torch
from tools import AudioCLIP

def d_clip_loss(x, y, use_cosine=False):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    if use_cosine:
        distance = 1 - (x @ y.t()).squeeze()
    else:
        distance = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    return distance


def audioclip_loss(x, y, model, use_scale=False, use_cosine=False):
    x = x / torch.linalg.norm(x, dim=-1, keepdim=True)
    y = y / torch.linalg.norm(y, dim=-1, keepdim=True)
    
    #aclp = AudioCLIP(pretrained=f'saved_ckpts/AudioCLIP-Full-Training.pt')
    scale_audio_image = torch.clamp(model.logit_scale_ai.exp(), min=1.0, max=100.0)
    
    if use_scale:
        distance = scale_audio_image * (1 -  x @ y.T)
    else:
        if use_cosine:
            distance = 1 -  x @ y.T
        else:
            distance = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
    
    return distance


def clip_loss(self, x_in, audio_embed):
    # x_in.shape(1,3,16,128,128)
    dists = 0

    if self.mask is not None:
        masked_input = x_in * self.mask
    else:
        masked_input = x_in
            
    if self.args.audio_emb_model == 'wav2clip':
        for idx in range(self.args.sequence_length):
            masked_frame = masked_input[:,:,idx,:,:]
            augmented_input = self.image_augmentations(masked_frame).add(1).div(2)
            clip_in = self.clip_normalize(augmented_input)
            image_embeds = self.clip_model.encode_image(clip_in).float()
            dist = d_clip_loss(image_embeds, audio_embed[idx,:], use_cosine=True) / self.args.sequence_length
            dists += dist.mean()
            
    elif self.args.audio_emb_model == 'audioclip':
        masked_frames = None        
        masked_input = masked_input.permute(0,2,1,3,4)
        masked_input = self.clip_normalize(masked_input)
        
        # resize every frame in a video. output shape: (1,16,3,224,224)
        for idx in range(masked_input.shape[1]):
            masked_frame = F.resize(masked_input[:,idx], [self.clip_size, self.clip_size])
            masked_frame = masked_frame.unsqueeze(1)
            if masked_frames == None:
                masked_frames = masked_frame
            else:
                masked_frames = torch.cat((masked_frames, masked_frame),1)
        
        # We want to sum over the averages, bs = 1, if change bs, need to modify the codes
        for bs in range(masked_frames.shape[0]):
            ((_, image_features, _), _), _ = self.audioclip(image=masked_frames[bs].cpu())       
            dist = audioclip_loss(audio_embed, image_features.cuda(), self.audioclip, use_scale=False, use_cosine=True) #(16,16)
            dist = torch.diag(dist).mean()
        
            dists += dist /  masked_frames.shape[0]

    return dists


def direction_loss(self, x, embed):      
    dists = 0
    
    if self.args.audio_emb_model == 'wav2clip':
        for idx in range(self.args.sequence_length-1):
            x2 = F.resize(x[:,:,idx+1,:,:], [self.clip_size, self.clip_size])
            x1 = F.resize(x[:,:,idx,:,:], [self.clip_size, self.clip_size])
            dis_x = self.clip_model.encode_image(x2).float() - self.clip_model.encode_image(x1).float()
            dis_embed = embed[idx+1] - embed[idx]
            dist = d_clip_loss(dis_x, dis_embed, use_cosine=True)
            dists += dist / (self.args.sequence_length - 1)
    elif self.args.audio_emb_model == 'audioclip':  
        frames = None 
        x = x.permute(0,2,1,3,4)
        x = self.clip_normalize(x)
        
        # get resized frames (bs, frames, channel, resolution, resolution)
        for idx in range(x.shape[1]):
            frame = F.resize(x[:,idx], [self.clip_size, self.clip_size])
            frame = frame.unsqueeze(1)
            if frames == None:
                frames = frame
            else:
                frames = torch.cat((frames, frame),1)
                
        # We want to sum over the averages, bs = 1, if change bs, need to modify the codes
        for bs in range(frames.shape[0]):   
            ((_, image_features, _), _), _ = self.audioclip(image=frames.squeeze().cpu()) #(16,1024)
            image_features = image_features.cuda()
            
            for idx in range(image_features.shape[0]-1):
                dis_x = image_features[idx+1]-image_features[idx]
                dis_embed = embed[idx+1] - embed[idx]               
                dist = audioclip_loss(dis_embed, dis_x, self.audioclip, use_scale=False, use_cosine=True)
                dists += dist / (image_features.shape[0] - 1)
    return dists
