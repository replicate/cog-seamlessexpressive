import contextlib
import logging
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from fairseq2.data import Collater, FileMapper
from fairseq2.data.audio import (
    AudioDecoder,
    WaveformToFbankConverter,
    WaveformToFbankOutput,
)
from torch import Tensor

from seamless_communication.cli.expressivity.evaluate.pretssel_inference_helper import (
    PretsselGenerator,
)
from seamless_communication.cli.m4t.evaluate.evaluate import (
    adjust_output_for_corrupted_inputs,
)

from seamless_communication.inference import (
    SequenceGeneratorOptions,
    BatchedSpeechOutput,
    Translator,
)
from seamless_communication.models.unity import (
    load_gcmvn_stats,
    load_unity_unit_tokenizer,
)
from seamless_communication.store import add_gated_assets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

from typing import Optional
import torch
import torchaudio
from cog import BasePredictor, Input, Path, BaseModel
import os

class ModelOutput(BaseModel):
    audio_out: Path
    text_out: str

LANG_TO_CODE = {
    "English": "eng",
    "French": "fra",
    "Spanish": "spa",
    "German": "deu",
    "Italian": "ita",
    "Chinese": "cmn",
}

class Predictor(BasePredictor):
    def setup(self) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32

        self.unit_tokenizer = load_unity_unit_tokenizer("seamless_expressivity")
        _gcmvn_mean, _gcmvn_std = load_gcmvn_stats("vocoder_pretssel")
        self.gcmvn_mean = torch.tensor(_gcmvn_mean, device=self.device, dtype=self.dtype)
        self.gcmvn_std = torch.tensor(_gcmvn_std, device=self.device, dtype=self.dtype)

        add_gated_assets(Path("./model/"))

        self.mapper = FileMapper(root_dir=".", cached_fd_count=10)
        self.decoder = AudioDecoder(dtype=torch.float32, device=self.device)
        self.wav_to_fbank = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15,
            channel_last=True,
            standardize=False,
            device=self.device,
            dtype=self.dtype,
        )

        self.collater = Collater(pad_value=0, pad_to_multiple=1)
        self.translator = Translator(
            "seamless_expressivity",
            vocoder_name_or_card=None,
            device=self.device,
        )

        self.text_generation_opts = SequenceGeneratorOptions(
            beam_size=5,
            soft_max_seq_len=(1,200),
            unk_penalty=torch.inf
        )
        self.unit_generation_opts = SequenceGeneratorOptions(
            beam_size=5,
            soft_max_seq_len=(25,50),
        )

        self.pretssel_generator = PretsselGenerator(
            "vocoder_pretssel",
            vocab_info=self.unit_tokenizer.vocab_info,
            device=self.device,
            dtype=self.dtype,
        )

    def normalize_fbank(self, data: WaveformToFbankOutput) -> WaveformToFbankOutput:
        fbank = data["fbank"]
        std, mean = torch.std_mean(fbank, dim=0)
        data["fbank"] = fbank.subtract(mean).divide(std)
        data["gcmvn_fbank"] = fbank.subtract(self.gcmvn_mean).divide(self.gcmvn_std)
        return data

    def predict(
        self,
        audio_in: Path = Input(
            description="Provide your input audio in your original language.",
            default=None,
        ),
        source_lang: str = Input(
            description="Provide the original language of the input audio.",
            default="English",
            choices=LANG_TO_CODE.keys(),
        ),
        target_lang: str = Input(
            description="Provide the target language for your output audio.",
            default="French",
            choices=LANG_TO_CODE.keys(),
        ),
        duration_factor: float = Input(
            description="Recommended: 1.0 for English, Mandarin, Spanish; 1.1 for German; 1.2 for French.",
            default=1.0
        ),
    ) -> ModelOutput:
        os.system(f"ffmpeg -i {audio_in} -acodec pcm_f32le -ar 16000 -ac 1 {audio_in.stem}.wav")
        audio_in = Path(f"{audio_in.stem}.wav")

        src_lang = LANG_TO_CODE[source_lang]
        tgt_lang = LANG_TO_CODE[target_lang]
        print("Preparing to convert from", src_lang, "to", tgt_lang)

        fbank = self.collater(
            self.normalize_fbank(
                self.wav_to_fbank(
                    self.decoder(
                        self.mapper(audio_in)["data"]
                    )
                )
            )
        )

        output_path = Path('./tmp') / audio_in.stem
        output_path.mkdir(parents=True, exist_ok=True)

        waveforms_dir = output_path / "waveform"
        waveforms_dir.mkdir(parents=True, exist_ok=True)

        hyps = []

        with contextlib.ExitStack() as stack:

            valid_sequences: Optional[Tensor] = None
            source = fbank["fbank"]

            # Skip corrupted audio tensors.
            valid_sequences = ~torch.any(
                torch.any(torch.isnan(source["seqs"]), dim=1), dim=1
            )

            if not valid_sequences.all():
                logger.warning(
                    f"Sample has some corrupted input."
                )
                source["seqs"] = source["seqs"][valid_sequences]
                source["seq_lens"] = source["seq_lens"][valid_sequences]

            # Skip performing inference when the input is entirely corrupted.
            if source["seqs"].numel() > 0:
                prosody_encoder_input = fbank["gcmvn_fbank"]
                text_output, unit_output = self.translator.predict(
                    source,
                    "s2st",
                    tgt_lang,
                    src_lang=src_lang,
                    text_generation_opts=self.text_generation_opts,
                    unit_generation_opts=self.unit_generation_opts,
                    unit_generation_ngram_filtering=False,
                    duration_factor=duration_factor,
                    prosody_encoder_input=prosody_encoder_input,
                )

                assert unit_output is not None
                speech_output = self.pretssel_generator.predict(
                    unit_output.units,
                    tgt_lang=tgt_lang,
                    prosody_encoder_input=prosody_encoder_input,
                )

            else:
                text_output = []
                speech_output = BatchedSpeechOutput(units=[], audio_wavs=[])

            if valid_sequences is not None and not valid_sequences.all():
                text_output, speech_output = adjust_output_for_corrupted_inputs(  # type: ignore[assignment]
                    valid_sequences,
                    text_output,
                    speech_output,
                )

            hyps += [str(s) for s in text_output]

            text = text_output[0]
            print(text)
            torchaudio.save(
                waveforms_dir / f"pred.wav",
                speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
                sample_rate=speech_output.sample_rate,
            )
            return ModelOutput(audio_out=waveforms_dir / f"pred.wav", text_out=str(text))
