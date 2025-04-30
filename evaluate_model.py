#!/usr/bin/env python
"""
Evaluate the harmonization model using two metrics:
1. MIDI conversion failure rate
2. Note distribution comparison between training set and generated results
"""

import os
import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from music21 import converter, note
from typing import List, Dict, Tuple

def count_midi_failures(output_dir: str) -> Tuple[int, int, float]:
    """
    Count how many MIDI files failed to be generated or are invalid.
    
    Args:
        output_dir: Directory containing the generated MIDI files
        
    Returns:
        Tuple of (total_files, failed_files, failure_percentage)
    """
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist")
        return 0, 0, 0
        
    total_files = 0
    failed_files = 0
    
    for filename in os.listdir(output_dir):
        if filename.endswith('.mid') or filename.endswith('.midi'):
            total_files += 1
            midi_path = os.path.join(output_dir, filename)
            try:
                # Try to parse the MIDI file
                score = converter.parse(midi_path)
                # Verify it has content
                if len(score.flat.notes) == 0:
                    failed_files += 1
                    print(f"Warning: Empty MIDI file: {filename}")
            except Exception as e:
                failed_files += 1
                print(f"Failed to process {filename}: {str(e)}")
    
    failure_percentage = (failed_files / total_files * 100) if total_files > 0 else 0
    return total_files, failed_files, failure_percentage

def get_note_distribution(data: List[List[int]] | List[List[List[int]]], voice_idx: int = None, is_atb: bool = False) -> Dict[int, float]:
    """
    Calculate the distribution of notes in the dataset.
    
    Args:
        data: List of sequences or a single sequence, where each sequence/chord is either [S,A,T,B] or [A,T,B]
        voice_idx: If provided, only analyze specific voice (0=S/A, 1=A/T, 2=T/B, 3=B)
        is_atb: Whether the data is in ATB format (True) or SATB format (False)
    
    Returns:
        Dictionary mapping MIDI note numbers to their frequency
    """
    if not data:
        print("Warning: Empty data provided to get_note_distribution")
        return {}
    
    # Check if data is a single sequence or list of sequences
    is_single_sequence = all(isinstance(x, int) for x in data[0])
    sequences = [data] if is_single_sequence else data
        
    notes = []
    for sequence in sequences:
        for chord in sequence:
            if voice_idx is not None:
                # Adjust voice index for ATB format
                adjusted_idx = voice_idx
                if is_atb:
                    if voice_idx == 0:  # Soprano -> Alto
                        adjusted_idx = 0
                    elif voice_idx == 1:  # Alto -> Tenor
                        adjusted_idx = 1
                    elif voice_idx == 2:  # Tenor -> Bass
                        adjusted_idx = 2
                    elif voice_idx == 3:  # Bass
                        adjusted_idx = 2  # Bass is the third voice in ATB format
                
                # Get the note if the index is valid
                if adjusted_idx < (3 if is_atb else 4):
                    notes.append(chord[adjusted_idx])
            else:
                notes.extend(chord)
    
    if not notes:
        print("Warning: No notes found in the data")
        return {}
    
    # Count occurrences
    counter = Counter(notes)
    total = sum(counter.values())
    
    # Convert to percentages
    return {note: count/total * 100 for note, count in counter.items()}

def midi_to_pitch(midi_number: int) -> str:
    """
    Convert MIDI number to pitch name (e.g., 60 -> 'C4', 61 -> 'C#4', etc.)
    MIDI number 60 corresponds to middle C (C4)
    """
    pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number - 12) // 12
    note = midi_number % 12
    return f"{pitch_names[note]}{octave}"

def plot_distribution_comparison(train_dist: Dict[int, float],
                               gen_dist: Dict[int, float],
                               title: str = "Note Distribution Comparison",
                               save_path: str = None):
    """
    Plot the comparison between training and generated note distributions.
    """
    if not train_dist and not gen_dist:
        print(f"Warning: No data to plot for {title}")
        return
        
    # Get all unique notes
    all_notes = sorted(set(list(train_dist.keys()) + list(gen_dist.keys())))
    
    if not all_notes:
        print(f"Warning: No notes found to plot for {title}")
        return
    
    # Prepare data for plotting
    x = np.arange(len(all_notes))
    train_values = [train_dist.get(note, 0) for note in all_notes]
    gen_values = [gen_dist.get(note, 0) for note in all_notes]
    
    # Convert MIDI numbers to pitch names
    pitch_names = [midi_to_pitch(note) for note in all_notes]
    
    # Create plot
    plt.figure(figsize=(15, 6))
    width = 0.35
    plt.bar(x - width/2, train_values, width, label='Training Data')
    plt.bar(x + width/2, gen_values, width, label='Generated Data')
    
    plt.xlabel('Pitch')
    plt.ylabel('Frequency (%)')
    plt.title(title)
    plt.legend()
    plt.xticks(x, pitch_names, rotation=45)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.close()

def test_midi_to_pitch():
    """
    Test the MIDI to pitch conversion function
    """
    test_cases = [
        (60, "C4"),  # Middle C
        (61, "C#4"),
        (72, "C5"),
        (48, "C3"),
        (69, "A4"),  # A440
        (57, "A3"),
        (45, "A2")
    ]
    
    print("\nTesting MIDI to pitch conversion:")
    for midi, expected in test_cases:
        result = midi_to_pitch(midi)
        print(f"MIDI {midi} -> {result} (Expected: {expected})")
        assert result == expected, f"Error: MIDI {midi} converted to {result}, expected {expected}"

def main():
    # Test MIDI to pitch conversion first
    test_midi_to_pitch()
    
    # Paths
    output_dir = "midi_outputs"  # Directory containing generated MIDI files
    training_data_path = "atb_parts.json"  # Path to training data
    results_dir = "evaluation_results"
    
    # List of model variants to evaluate
    model_variants = [
        ("lstm_nopen_1", "LSTM (No Penalty) v1"),
        ("lstm_nopen_2", "LSTM (No Penalty) v2"),
        ("lstm_nopen_3", "LSTM (No Penalty) v3"),
        ("lstm_pen_1", "LSTM (With Penalty) v1"),
        ("lstm_pen_2", "LSTM (With Penalty) v2"),
        ("lstm_pen_3", "LSTM (With Penalty) v3"),
        ("transformer_nopen_1", "Transformer (No Penalty) v1"),
        ("transformer_nopen_2", "Transformer (No Penalty) v2"),
        ("transformer_nopen_3", "Transformer (No Penalty) v3"),
        ("transformer_pen_1", "Transformer (With Penalty) v1"),
        ("transformer_pen_2", "Transformer (With Penalty) v2"),
        ("transformer_pen_3", "Transformer (With Penalty) v3")
    ]
    
    print("Starting evaluation...")
    print(f"Training data path: {training_data_path}")
    
    # Create results directory
    try:
        os.makedirs(results_dir, exist_ok=True)
        print(f"Created results directory: {results_dir}")
    except Exception as e:
        print(f"Error creating results directory: {str(e)}")
        return
    
    # 1. Evaluate MIDI conversion failures
    print("\nChecking MIDI files...")
    total, failed, failure_rate = count_midi_failures(output_dir)
    print(f"\nMIDI Conversion Results:")
    print(f"Total files: {total}")
    print(f"Failed files: {failed}")
    print(f"Failure rate: {failure_rate:.2f}%")
    
    # Save MIDI conversion results
    results_file = os.path.join(results_dir, "midi_conversion_results.txt")
    with open(results_file, 'w') as f:
        f.write("MIDI Conversion Evaluation Results\n")
        f.write("================================\n\n")
        f.write(f"Total files analyzed: {total}\n")
        f.write(f"Failed files: {failed}\n")
        f.write(f"Failure rate: {failure_rate:.2f}%\n")
    
    print(f"\nResults saved to {results_file}")
    
    # 2. Compare note distributions
    print("\nLoading training data...")
    try:
        with open(training_data_path, 'r') as f:
            training_data = json.load(f)
        print(f"Loaded training data with {len(training_data)} sequences")
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        return
    
    # Calculate training distributions
    print("\nCalculating training distributions...")
    train_dist_all = get_note_distribution(training_data, is_atb=True)
    train_dist_bass = get_note_distribution(training_data, voice_idx=3, is_atb=True)
    
    # Evaluate each model variant
    for model_id, model_name in model_variants:
        print(f"\nEvaluating {model_name}...")
        
        # Load generated results
        pred_file = f"Junhan Cui/OUTPUT/satb_predictions_{model_id}.json"
        try:
            with open(pred_file, 'r') as f:
                generated_data = json.load(f)
            print(f"Loaded {len(generated_data)} sequences from {model_id}")
        except Exception as e:
            print(f"Error loading generated results for {model_id}: {str(e)}")
            continue
        
        if not generated_data:
            print(f"Warning: No generated data found for {model_id}")
            continue
        
        # Calculate generated distributions
        gen_dist_all = get_note_distribution(generated_data, is_atb=False)
        gen_dist_bass = get_note_distribution(generated_data, voice_idx=3, is_atb=False)
        
        # Plot overall distribution
        if train_dist_all and gen_dist_all:
            plot_distribution_comparison(
                train_dist_all,
                gen_dist_all,
                title=f"Overall Note Distribution - {model_name}",
                save_path=os.path.join(results_dir, f"overall_distribution_{model_id}.png")
            )
        
        # Plot bass distribution
        if train_dist_bass and gen_dist_bass:
            plot_distribution_comparison(
                train_dist_bass,
                gen_dist_bass,
                title=f"Bass Voice Note Distribution - {model_name}",
                save_path=os.path.join(results_dir, f"bass_distribution_{model_id}.png")
            )

if __name__ == "__main__":
    main() 
