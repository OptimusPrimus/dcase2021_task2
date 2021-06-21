import torch
from common.architectures.cp_resnet import get_model_based_on_rho, maxrf


class ResNet(torch.nn.Module):

    def __init__(
            self,
            input_shape,
            n_classes=1,
            base_channels=128,
            rho_t=5,
            rho_f=5,
            **kwargs
    ):
        super().__init__()

        input_shape = (1, *input_shape)

        self.net = get_model_based_on_rho(
            rho_t=rho_t,
            rho_f=rho_f,
            base_channels=base_channels,
            blocks='444',
            n_classes=n_classes,
            arch="cp_speech_resnet",
            config_only=False,
            input_shape=input_shape
        )

        print(f"Maximum receptive field: {maxrf(24)}")

    def forward(self, batch):
        x = batch['input']
        batch['logits'], batch['embedding'] = self.net(x)
            #.view(-1, self.num_outputs)
        return batch
