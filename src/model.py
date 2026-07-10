import torch
import torch.nn as nn


class DeepfakeAudioCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Conv1d(
                1,
                16,
                kernel_size=3
            ),

            nn.ReLU(),

            nn.MaxPool1d(2),


            nn.Conv1d(
                16,
                32,
                kernel_size=3
            ),

            nn.ReLU(),

            nn.MaxPool1d(2)
        )


        self.fc = nn.Sequential(

            nn.Linear(32 * 8, 64),

            nn.ReLU(),

            nn.Linear(64,2)

        )


    def forward(self,x):

        x=self.network(x)

        x=x.view(x.size(0),-1)

        x=self.fc(x)

        return x