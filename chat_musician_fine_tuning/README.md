# MIDI Harmonization with ChatMusician LoRA Fine‑Tuning
A pipeline for fine‑tuning the ChatMusician (LLaMA‑2‑7B) model with the LoRA/LoKr adapter on SATB harmonization tasks using JSB Chorales and CoCoChorales datasets.

## Directory Structure
```
.
├── .gitignore
├── data
│   ├── cocochorales        # coco-full midi files
│   ├── dataset_jsb         # incomplete, for LoKr, better aug
│   ├── dataset_jsb_abc             ## dataset used for LoRA
│   ├── download_cocochorales.py
│   ├── jsb-chorales-quarter.pkl    ## raw jsb dataset from the official repo
│   ├── parse_midi.py       # helper for coco
│   ├── preprocess_coco.py  # incomplete attempt
│   ├── preprocess_jsb.py   # incomplete attempt
│   └── preprocess_jsb_abc.py       ## preprocessing script used for LoRA
├── fine_tune_lokr.py       # incomplete attempt
├── fine_tune_lora.py               ## working script for fine tuning
├── inference_lora.py               ## working script for inference
├── lora-clean                      ## checkpoints
├── lora-final                      ## final weights
├── loss_curve_lora.png             ## training loss
├── parse_abc.py                    ## post-processing, but model output is not consistent enough
├── requirements.txt                ## dependencies other than torch
├── setup.sh                        ## intended for my env only
└── train_history_lora.json         ## training history
```

## Requirements
```bash
pip install -r requirements.txt
```
> Data processing uses my custom library [pyvoicing](https://pypi.org/project/pyvoicing/0.1.3/).

## Finetuning
```bash
python fine_tune_lora.py
```
> Be aware that the pretrained weights are quite large. Make sure you have at least 16 GB of free disk space for the FP16 model.

## Generation
```bash
python inference_lora.py
```
> Modify the melody/prompt inside the script.
