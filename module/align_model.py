import torch
from torch import nn
import whisper

   
class RNN(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size,
        output_size,
        num_layers: int=2,
        dropout: float=0.1,
        batch_first: bool=True, 
        bidirectional: bool=True,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=batch_first,
                            bidirectional=bidirectional)
        self.relu = nn.Mish()
        self.fc = nn.Linear(hidden_size + (bidirectional * hidden_size), 
                            output_size + 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.relu(out)
        out = self.fc(out)

        return out

class AlignModel(torch.nn.Module):
    def __init__(self,
        whisper_model: whisper.Whisper,
        embed_dim: int=1280,
        hidden_dim: int=512,
        dropout: float=0.2,
        output_dim: int=10000,
        freeze_encoder: bool=False,
        train_alignment: bool=True,
        train_transcribe: bool=False,
        device: str='cuda'
        ) -> None:
        super().__init__()
        self.whisper_model = whisper_model
        
        # Text Alignment
        self.align_rnn = RNN(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            output_size=output_dim,
                            dropout=dropout)

        self.freeze_encoder = freeze_encoder

        self.train_alignment = train_alignment
        self.train_transcribe = train_transcribe

        self.device = device

    def forward(self, x, y_in=None):
        # x => Mel
        # y_in => whisper decoder input
        # You can ignore y_in if you are doing alignment task
        if self.freeze_encoder:
            with torch.no_grad():
                embed = self.whisper_model.embed_audio(x)
        else:
            embed = self.whisper_model.embed_audio(x)

        # Align Logit
        align_logit = None
        if self.train_alignment:
            align_logit = self.align_rnn(embed)

        # Transcribe Logit
        transcribe_logit = None
        if self.train_transcribe:
            assert y_in is not None
            transcribe_logit = self.whisper_model.logits(tokens=y_in,
                                                         audio_features=embed)
        

        return align_logit, transcribe_logit