import torch.nn as nn
import torch
import laion_clap
from .build import BACKBONE_REGISTRY
from .backbone import Backbone


class CLAP(Backbone):
    def __init__(self, pretrained_path, device):
        super().__init__()
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt(pretrained_path)  # Load CLAP checkpoint

        self.device = device
        self.model.to(device)
        self.model.eval()

        # Freeze everything
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        # Audio and text output dimensions
        self.audio_out_features = 512  # CLAP's audio embedding size
        self.text_out_features = 512  # CLAP's text embedding size

    def forward_audio(self, audio_waveform):
        """
        Extract audio embeddings.
        :param audio_waveform: Tensor of shape (batch, time)
        :return: Audio embeddings (batch, 512)
        """
        with torch.no_grad():
            audio_embeddings = self.model.get_audio_embedding_from_data(
                x=audio_waveform, use_tensor=True
            )
        return audio_embeddings

    def forward_text(self, text_prompts):
        """
        Extract text embeddings.
        :param text_prompts: List of text prompts
        :return: Text embeddings (batch, 512)
        """
        with torch.no_grad():
            text_embeddings = self.model.get_text_embedding(
                text_prompts, use_tensor=True
            )
        return text_embeddings


@BACKBONE_REGISTRY.register()
def clap_backbone(device, pretrained_path=""):
    model = CLAP(pretrained_path, device)
    return model

