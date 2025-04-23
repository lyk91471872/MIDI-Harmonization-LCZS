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

def get_note_distribution(data: List[List[List[int]]], voice_idx: int = None, is_atb: bool = False) -> Dict[int, float]:
    """
    Calculate the distribution of notes in the dataset.
    
    Args:
        data: List of sequences, where each sequence is a list of chords, and each chord is a list of [S,A,T,B] or [A,T,B] notes
        voice_idx: If provided, only analyze specific voice (0=S/A, 1=A/T, 2=T/B, 3=B)
        is_atb: Whether the data is in ATB format (True) or SATB format (False)
    
    Returns:
        Dictionary mapping MIDI note numbers to their frequency
    """
    if not data:
        print("Warning: Empty data provided to get_note_distribution")
        return {}
        
    notes = []
    for sequence in data:
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
                if adjusted_idx < len(chord):
                    notes.append(chord[adjusted_idx])
                else:
                    print(f"Warning: Voice index {adjusted_idx} out of range for chord {chord}")
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
    
    # Create plot
    plt.figure(figsize=(15, 6))
    width = 0.35
    plt.bar(x - width/2, train_values, width, label='Training Data')
    plt.bar(x + width/2, gen_values, width, label='Generated Data')
    
    plt.xlabel('MIDI Note Number')
    plt.ylabel('Frequency (%)')
    plt.title(title)
    plt.legend()
    plt.xticks(x, all_notes, rotation=45)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.close()

def main():
    # Paths
    output_dir = "midi_outputs"  # Directory containing generated MIDI files
    training_data_path = "atb_parts.json"  # Path to training data
    generated_data_path = "OUTPUT/satb_predictions.json"  # Path to generated SATB predictions
    results_dir = "evaluation_results"
    
    print("Starting evaluation...")
    print(f"Output directory: {output_dir}")
    print(f"Training data path: {training_data_path}")
    print(f"Generated data path: {generated_data_path}")
    
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
    
    # 2. Compare note distributions
    print("\nLoading training data...")
    try:
        with open(training_data_path, 'r') as f:
            training_data = json.load(f)
        print(f"Loaded training data with {len(training_data)} sequences")
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        return
    
    # Load generated results
    print("\nLoading generated results...")
    try:
        with open(generated_data_path, 'r') as f:
            generated_data = json.load(f)
        print(f"Loaded {len(generated_data)} sequences")
    except Exception as e:
        print(f"Error loading generated results: {str(e)}")
        return
    
    if not generated_data:
        print("Warning: No generated data found")
        return
    
    # Calculate distributions
    print("\nCalculating distributions...")
    voice_names = ['Soprano', 'Alto', 'Tenor', 'Bass']
    for i in range(4):  # For each voice
        print(f"\nProcessing {voice_names[i]} voice...")
        train_dist = get_note_distribution(training_data, i, is_atb=True)
        gen_dist = get_note_distribution(generated_data, i, is_atb=False)
        
        if not train_dist or not gen_dist:
            print(f"Warning: No data found for {voice_names[i]} voice")
            continue
        
        # Plot comparison
        plot_distribution_comparison(
            train_dist, 
            gen_dist,
            title=f"{voice_names[i]} Voice Note Distribution",
            save_path=os.path.join(results_dir, f"{voice_names[i].lower()}_distribution.png")
        )
    
    # Also plot overall distribution
    print("\nCalculating overall distribution...")
    train_dist_all = get_note_distribution(training_data, is_atb=True)
    gen_dist_all = get_note_distribution(generated_data, is_atb=False)
    
    if train_dist_all and gen_dist_all:
        plot_distribution_comparison(
            train_dist_all,
            gen_dist_all,
            title="Overall Note Distribution",
            save_path=os.path.join(results_dir, "overall_distribution.png")
        )
    else:
        print("Warning: No data found for overall distribution")

if __name__ == "__main__":
    main() 