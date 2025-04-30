# MIDI Harmonization LCZS


# This is Junhan Cui
    my role in this team is building a dual model training setup, and training it with small dataset, compare the performance between models and w & wo penalty.
    ## 1. Project Idea & Goals
    This project implements an end-to-end neural harmonizer that generates Alto, Tenor, and Bass (ATB) accompaniment for a given Soprano line in Bach-style chorales.  
    - **Goals:**  
      1. Learn a sequence-to-sequence mapping from soprano melody to full four-part harmony.  
      2. Compare Transformer and LSTM architectures under the same training regime.  
      3. Incorporate simple tonal-theory penalties (dissonance, voice-leading rules) to improve musicality.  
      4. Automatically export generated harmonies as both JSON and MIDI files.

    ## 2. Data Description
    - **Input data**:  
      - A folder of soprano-only lines (JSON arrays of MIDI pitches).  
    - **Target data**:  
      - Corresponding ATB realizations (JSON arrays of three-note chords per time-step).  
    - **Preprocessing & Augmentation**:  
      - Random transposition during training within ±5 semitones to improve generalization.  
      - Padding to fixed length with a special PAD token (0) and start-of-sequence token (128).

    ## 3. Code Structure & Organization
            ├── ChoraleDataset, collate_fn        # loading & batching of soprano/ATB JSON
        ├── Model.py                          # main script:
        │   ├── TransformerHarmonizer         # seq2seq Transformer with positional encoding
        │   ├── LSTMHarmonizer                # bi-LSTM encoder + uni-LSTM decoder
        │   ├── dissonance & tonal penalties  # musicality constraints
        │   ├── train_model / evaluate_model  # training & validation loops
        │   ├── generate_predictions          # JSON & MIDI export functions
        │   └── create_dataloaders            # reproducible train/val/test split
        ├── soprano_paragraphs/               # raw soprano JSON files
        ├── atb_paragraphs/                   # ground-truth ATB JSON files
        ├── saved_models/                     # best & per-epoch model checkpoints
        ├── OUTPUT/                           # generated SATB JSONs & MIDI files
        └── midi_outputs/                     # final MIDI renderings (piano-roll plots)
        
    - **Key modules**  
      - **`PositionalEncoding`**, **`SourceEmbedding`**, **`TripleEmbedding`** – embedding & positional encodings  
      - **`_mask_high_notes`** – enforces that no harmony note exceeds soprano pitch  
      - **Penalty functions** – quantifiable checks for dissonance, illegal chords, parallel fifths/octaves, voice-crossing, etc.  
      - **`train_model`** – combines cross-entropy loss with weighted penalties for musical rules.  

            if you want to run my model----

            1: click into folder with my name
            2: open "model.py"
            3: run the code

            after running the code, the termnal will ask you some questions----
            1: which model you want to use (transformer or LSTM)
            2: do penalty involve during this training

            The pridiction result will appeared at OUTPUT folder~
            Enjoy your training!

# Jiyeoung_Sim_Evaluation for JunHan's model
I was responsible for the evaluation of Junhan's harmonization model. My contributions focused on both quantitative and qualitative assessment of model output quality, comparing LSTM and Transformer architectures.

## Running the Evaluation
To evaluate the models, run:
```bash
python3.11 evaluate_model.py
```

The evaluation script performs the following steps:
1. Tests MIDI to pitch name conversion accuracy (e.g., MIDI 60 → C4)
2. Analyzes MIDI conversion quality in the `midi_outputs` directory
3. Loads and processes:
   - Training data from `atb_parts.json`
   - Model predictions from `OUTPUT/satb_predictions_{model_id}.json`
4. Generates distribution plots comparing training vs. generated data

## Latest Evaluation Results

### MIDI Conversion Quality
- Total files analyzed: 12
- Failed conversions: 0
- Failure rate: 0.00%

### Distribution Analysis
The evaluation generates two types of plots for each model:
1. **Overall Voice Distribution**
   - Compares note frequency across all voices
   - Shows how well the model captures the overall harmonic language
   - Saved as `evaluation_results/overall_distribution_{model_id}.png`

2. **Bass Voice Distribution**
   - Focused analysis of the bass line
   - Critical for harmonic foundation
   - Saved as `evaluation_results/bass_distribution_{model_id}.png`

### Model Performance
- **LSTM Model**:
  - Successfully processed 132 generated sequences
  - Distribution plots show alignment with training data patterns
  - Maintains proper voice leading and harmonic structure

### Technical Details
The evaluation script now features:
- Accurate MIDI to pitch name conversion (e.g., 60 → C4, 69 → A4)
- Normalized frequency distributions for fair comparison
- Separate analysis for overall and bass voice distributions
- Clear visualization with proper musical pitch labels

### Key Findings
1. **MIDI Generation**: Perfect conversion rate (0% failures)
2. **Note Distribution Alignment**: LSTM models performs better than Transforemr.
3. **Bass Voice Analysis**: LSTM with penalty has a minor impact.
4. **Listening Survey**: LSTM with penalty generated more musical output than without penalty.

## Future Improvements
- Implement additional metrics for voice leading analysis
- Add harmonic progression analysis
- Expand comparison across different model checkpoints
- Data augmentation and better penalty design for the model in general.

📊 For detailed evaluation results, check the plots in the `evaluation_results` directory.
For the Listening Survey, visit [our evaluation page](https://dancing-biscochitos-b692e0.netlify.app/)

# Ao_Zhang_LSTM

Model:"lstm_epoch14.pth"
Run your code at Terminal

1. Data preperation:
-Run "build_dataset.py" to get your own datasets, or you can use mine: under 'data' folder, there are .npz files compressed from Coco_Chorales_Tiny dataset.
2. Training:
-Run "train.py" to continue training, you can stop at any epoch/checkpoint. 
3. MIDI prediction:
-Run "Inference.py" to predict, be careful about your dataset path.
4. Evaluation:
-Run "eval.py" to evaluate your model, you will get Top1 and Top5 score to see how the ATB parts perform.

# Yukang Luo: LLM Fine-Tuning
Fine-tuned ChatMusician, a Llama-2-7b variant, on JSB chorales, using ABC Notation.

## File Structure
See [chat_musician_fine_tuning/README.md](https://github.com/lyk91471872/MIDI-Harmonization-LCZS/blob/main/chat_musician_fine_tuning/README.md) for details.

## Progress
### Completed
* Fine-tuning with LoRA using PEFT.

### Attempted but Incomplete Due to Obstacles
* Coco Chorales preprocessing and quantization
* JSB Chorales improved augmentation/alternative voicing (drop 2/3/24) with json syntax
