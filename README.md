# MIDI Harmonization LCZS
this is readme~

## some subtitle
* some key point
> some quote

# This is Junhan Cui
my role in this team is building a dual model training setup, and training it with small dataset, compare the performance between models and w & wo penalty.

if you want to run my model----

1: click into folder with my name
2: open "model.py"
3: run the code

after running the code, the termnal will ask you some questions----
1: which model you want to use (transformer or LSTM)
2: do penalty involve during this training

The pridiction result will appeared at OUTPUT folder~
Enjoy your training!

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
