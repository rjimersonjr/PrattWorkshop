import nemo
# NeMo's ASR collection - this collections contains complete ASR models and
# building blocks (modules) for ASR
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from ruamel.yaml import YAML
from omegaconf import DictConfig
import copy, json, numpy, time
from pyctcdecode import build_ctcdecoder

from dataclasses import dataclass, field, is_dataclass
from nemo.collections.asr.parts.submodules import ctc_beam_decoding
from jiwer import wer, cer
from omegaconf import DictConfig

import torch
import torch.nn as nn

def enable_bn_se(m):
    if type(m) == nn.BatchNorm1d:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

    if 'SqueezeExcite' in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

train_manifest = "/content/gdrive/MyDrive/PrattWorkshop/Training_data.json"
test_manifest = "/content/gdrive/MyDrive/PrattWorkshop/Test_data.json"

config_path = '/content/gdrive/MyDrive/PrattWorkshop/configs/config.yaml'
yaml = YAML(typ='safe')
with open(config_path) as f:
    params = yaml.load(f)

params['model']['train_ds']['manifest_filepath'] = train_manifest
params['model']['validation_ds']['manifest_filepath'] = test_manifest
params['model']['validation_ds']['batch_size'] = 8

params['model']['labels'] = ["'","a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

freeze_encoder = True

if freeze_encoder:
    asr_model.encoder.freeze()
    asr_model.encoder.apply(enable_bn_se)
    print("Model encoder has been frozen, and batch normalization has been unfrozen")

# Point to the data we'll use for fine-tuning as the training set
asr_model.setup_training_data(train_data_config=params['model']['train_ds'])

# Point to the new validation data for fine-tuning
asr_model.setup_validation_data(val_data_config=params['model']['validation_ds'])

trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=5)
trainer.fit(asr_model)

asr_model.save_to("/content/gdrive/MyDrive/PrattWorkshop/asr_50_freeze.nemo")

# Bigger batch-size = bigger throughput
params['model']['validation_ds']['batch_size'] = 16

# Setup the test data loader and make sure the model is on GPU
asr_model.setup_test_data(test_data_config=params['model']['validation_ds'])
asr_model.cuda()
asr_model.eval()

# We will be computing Word Error Rate (WER) metric between our hypothesis and predictions.
# WER is computed as numerator/denominator.
# We'll gather all the test batches' numerators and denominators.
wer_nums = []
wer_denoms = []

# Loop over all test batches.
# Iterating over the model's `test_dataloader` will give us:
# (audio_signal, audio_signal_length, transcript_tokens, transcript_length)
# See the AudioToCharDataset for more details.
for test_batch in asr_model.test_dataloader():
        test_batch = [x.cuda() for x in test_batch]
        targets = test_batch[2]
        targets_lengths = test_batch[3]
        log_probs, encoded_len, greedy_predictions = asr_model(
            input_signal=test_batch[0], input_signal_length=test_batch[1]
        )
        # Notice the model has a helper object to compute WER
        asr_model.wer.update(predictions=greedy_predictions, predictions_lengths=None, targets=targets, targets_lengths=targets_lengths)
        _, wer_num, wer_denom = asr_model.wer.compute()
        asr_model.wer.reset()
        wer_nums.append(wer_num.detach().cpu().numpy())
        wer_denoms.append(wer_denom.detach().cpu().numpy())

        # Release tensors from GPU memory
        del test_batch, log_probs, targets, targets_lengths, encoded_len, greedy_predictions

# We need to sum all numerators and denominators first. Then divide.
print(f"WER = {sum(wer_nums)/sum(wer_denoms)}")

