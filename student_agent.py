import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import torchvision.transforms.functional as TF
from PIL import Image
# Do not modify the input of the 'act' function and the '__init__' function. 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PerturbedLinear(nn.Module):
    def __init__(self, input_features, output_features, noise_scale=2.5):
        # 使用直接調用父類的方式初始化
        nn.Module.__init__(self)
        self.in_features = input_features
        self.out_features = output_features
        self.noise_scale = noise_scale

        # 均值參數
        self.weight_mean = nn.Parameter(torch.empty(output_features, input_features))
        self.bias_mean   = nn.Parameter(torch.empty(output_features))

        # 標準差參數
        self.weight_std  = nn.Parameter(torch.empty(output_features, input_features))
        self.bias_std    = nn.Parameter(torch.empty(output_features))

        # 隨機噪音張量（不參與梯度計算）
        self.register_buffer("weight_noise", torch.empty(output_features, input_features))
        self.register_buffer("bias_noise",   torch.empty(output_features))

        self._initialize_weights()
        self.sample_noise()

    def _initialize_weights(self):
        limit = (self.in_features) ** (-0.5)
        nn.init.uniform_(self.weight_mean, -limit, limit)
        nn.init.constant_(self.weight_std, self.noise_scale * limit)
        nn.init.uniform_(self.bias_mean, -limit, limit)
        nn.init.constant_(self.bias_std, self.noise_scale * limit)

    @staticmethod
    def _transform_noise(tensor):
        # Sign × √abs
        return tensor.sign() * tensor.abs().sqrt()

    def sample_noise(self):
        # 產生獨立的輸入/輸出噪音向量
        eps_in  = self._transform_noise(torch.randn(self.in_features, device=self.weight_noise.device))
        eps_out = self._transform_noise(torch.randn(self.out_features, device=self.weight_noise.device))
        
        self.weight_noise.copy_(eps_out.unsqueeze(1) @ eps_in.unsqueeze(0))
        self.bias_noise.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mean 
            w_noise = self.weight_std * self.weight_noise
            w = w + w_noise
            b = self.bias_mean 
            b_noise = self.bias_std * self.bias_noise
            b = b + b_noise
        else:
            w, b = self.weight_mean, self.bias_mean
        return torch.addmm(b, x, w.t())

class DuelingNetwork(nn.Module):
    def __init__(self, channels: int, num_actions: int):
        super(DuelingNetwork, self).__init__()
        # Convolutional feature extractor
        self._conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=False),
            nn.Flatten()
        )
        flat_size = 3136
        self._value_hidden = PerturbedLinear(flat_size, 512)
        self._adv_hidden = PerturbedLinear(flat_size, 512)
        self._flatten = nn.Flatten()
        self._value_out = PerturbedLinear(512, 1)
        self._adv_out = PerturbedLinear(512, num_actions)
        self.ReLU = nn.ReLU(inplace=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = x.float() / 255.0

        embeddings = self._conv(x)
    
        # value branch
        v_hidden = self._value_hidden(embeddings)
        v_hidden = self.ReLU(v_hidden)
        value = self._value_out(v_hidden)
    
        # advantage branch
        a_hidden = self._adv_hidden(embeddings)
        a_hidden = self.ReLU(a_hidden)
        advantage = self._adv_out(a_hidden)
    
        # center advantage and combine with value
        advantage = advantage - advantage.mean(dim=1, keepdim=True)
        q_values = value + advantage
    
        return q_values

    def reset_noise(self) -> None:
        perturbed_linears = (self._value_hidden, self._value_out, self._adv_hidden, self._adv_out)
        for layer in perturbed_linears:
            layer.sample_noise()
model_path = 'the_final_ckpt.pth'
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.device = DEVICE
        self.action_space = gym.spaces.Discrete(12)
        self.online = DuelingNetwork(4, 12).to(self.device)
        self.ckpt = torch.load(model_path, map_location=self.device)
        self.online.load_state_dict(self.ckpt['model'])
        self.skip_frame = 4
        self.skip_count = 0
        self.last_action = None 
        self.first = True
        self.frames = deque(maxlen=4)
    def observation(self, obs):
        # Convert array to PIL image
        image = Image.fromarray(obs)

        # Convert to grayscale
        gray_img = image.convert("L")  # "L" = grayscale mode

        # Resize to target dimensions
        resized = gray_img.resize((90, 84), resample=Image.BILINEAR)

        # Convert to tensor (auto scales to [0,1] and shape C×H×W)
        tensor = TF.to_tensor(resized)  # shape: (1, 84, 90), dtype=float32

        return tensor.numpy()
    def act(self, observation):
        self.online.reset_noise()
        obs = self.observation(observation)
        if self.first:
           self.frames.clear()
           for _ in range(4): self.frames.append(obs)
           self.first = False 
        if self.skip_count != 0:
            self.skip_count += 1
            self.skip_count %= self.skip_frame
            return self.last_action
        else:
            self.skip_count += 1
            self.frames.append(obs)
            with torch.no_grad():
                stacked_frames = torch.stack(list(self.frames), dim=0).to(self.device)
                stacked_frames = np.transpose(stacked_frames, (1, 0, 2, 3))  # Shape: (4, 84, 84)
                q_values = self.online(stacked_frames)
                action = q_values.max(1)[1].item()
            self.last_action = action
            return action

