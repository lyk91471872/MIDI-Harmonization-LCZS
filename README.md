# MIDI Harmonization LCZS
this is readme~

## some subtitle
* some key point
> some quote

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
I was responsible for the evaluation of Junhan’s harmonization model. My contributions focused on both quantitative and qualitative assessment of model output quality, comparing Transformer and LSTM architectures with and without tonal-theory penalties.
For the evaluation, you need to run this command: python 3.11 evaluate_model.py
This will give you, 
1) First test the MIDI to pitch conversion functionality
2) Check for any MIDI conversion failures in the midi_outputs directory
3) Load the training data from atb_parts.json
4) Load the generated predictions from OUTPUT/satb_predictions.json
5) Calculate note distributions for both training and generated data
6) Create comparison plots for:
- Overall note distribution (all voices)
- Bass voice distribution

7) Save all plots in the evaluation_results directory
The results will include:
- A text file with MIDI conversion statistics (evaluation_results/midi_conversion_results.txt)
- Three distribution comparison plots:
- evaluation_results/overall_distribution.png
- evaluation_results/bass_distribution.png

🔍 Evaluation Components:
	•	evaluation_model.py: This script computes model performance by comparing the note distribution of predicted ATB voices to the training set and measuring MIDI conversion failure rate. Further, it allows you to compare the bass note distribution with the training data, which is the most important voice in 4-voices harmony composition, about the LSTM model with and without penalty,
	•	Overall Note Distribution Analysis: Histograms comparing the frequency of MIDI pitches in the training data versus predicted data for all voice parts. More it resembles the training data, better the result.
	•	Bass Note Distribution: A focused analysis on bass voice accuracy, highlighting pitch alignment.
	•	MIDI Conversion Failure Rate: Ensured robustness of model output; Our models produced valid MIDI without conversion errors.
	•	Listening Assessment: Designed and conducted a subjective listening test using a survey. Participants rated the musical quality of generated harmonizations. 
  📊 For the Listening Survey, you can check HERE! (https://dancing-biscochitos-b692e0.netlify.app/) 

The results confirmed that LSTM with penalties yielded the most stylistically faithful and musically pleasing outputs, as supported by both statistical distribution alignment and listener feedback.

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
