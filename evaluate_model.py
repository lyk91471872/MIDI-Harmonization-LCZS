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
    results_dir = "evaluation_results"
    
    print("Starting MIDI conversion evaluation...")
    print(f"Output directory: {output_dir}")
    
    # Create results directory
    try:
        os.makedirs(results_dir, exist_ok=True)
        print(f"Created results directory: {results_dir}")
    except Exception as e:
        print(f"Error creating results directory: {str(e)}")
        return
    
    # Evaluate MIDI conversion failures
    print("\nChecking MIDI files...")
    total, failed, failure_rate = count_midi_failures(output_dir)
    print(f"\nMIDI Conversion Results:")
    print(f"Total files: {total}")
    print(f"Failed files: {failed}")
    print(f"Failure rate: {failure_rate:.2f}%")
    
    # Save results to a text file
    results_file = os.path.join(results_dir, "midi_conversion_results.txt")
    with open(results_file, 'w') as f:
        f.write("MIDI Conversion Evaluation Results\n")
        f.write("================================\n\n")
        f.write(f"Total files analyzed: {total}\n")
        f.write(f"Failed files: {failed}\n")
        f.write(f"Failure rate: {failure_rate:.2f}%\n")
    
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main() 
