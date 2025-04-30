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
from typing import List, Dict, Tuple, Any

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

def calculate_note_distribution(data: List[Any], voice: str = None, is_training: bool = False) -> Dict[int, int]:
    """Calculate note distribution from the data."""
    distribution = {}
    
    if is_training:
        # Training data format: List[List[List[int]]] where inner lists are [soprano, alto, tenor, bass]
        for sequence in data:
            for chord in sequence:
                if voice == 'bass':
                    midi_number = chord[-1]  # Last note is bass
                    distribution[midi_number] = distribution.get(midi_number, 0) + 1
                else:  # All voices
                    for midi_number in chord:
                        distribution[midi_number] = distribution.get(midi_number, 0) + 1
    else:
        # Generated data format: List[Dict] with 'notes' field containing voice information
        for item in data:
            notes = item['notes']
            if voice:
                # Filter notes for the specified voice
                notes = [note for note in notes if note['voice'] == voice]
            for note in notes:
                midi_number = note['midi_number']
                distribution[midi_number] = distribution.get(midi_number, 0) + 1
    
    return distribution

def plot_distribution_comparison(
    training_dist: Dict[int, int],
    generated_dist: Dict[int, int],
    title: str,
    output_path: str
):
    """Plot comparison of note distributions."""
    if not training_dist and not generated_dist:
        print(f"Warning: No data to plot for {title}")
        return

    # Convert MIDI numbers to pitch names
    training_pitches = {midi_to_pitch(k): v for k, v in training_dist.items()}
    generated_pitches = {midi_to_pitch(k): v for k, v in generated_dist.items()}

    # Get all unique pitches
    all_pitches = sorted(set(list(training_pitches.keys()) + list(generated_pitches.keys())))

    # Prepare data for plotting
    training_values = [training_pitches.get(pitch, 0) for pitch in all_pitches]
    generated_values = [generated_pitches.get(pitch, 0) for pitch in all_pitches]

    # Normalize the distributions
    training_total = sum(training_values)
    generated_total = sum(generated_values)
    
    if training_total > 0:
        training_values = [v/training_total for v in training_values]
    if generated_total > 0:
        generated_values = [v/generated_total for v in generated_values]

    plt.figure(figsize=(15, 6))
    width = 0.35

    plt.bar(x - width/2, training_values, width, label='Training Data')
    plt.bar(x + width/2, generated_values, width, label='Generated Data')

    plt.xlabel('Pitch')
    plt.ylabel('Normalized Frequency')
    plt.title(title)
    plt.xticks(x, all_pitches, rotation=45)
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
    training_dist_all = calculate_note_distribution(training_data, is_training=True)
    training_dist_bass = calculate_note_distribution(training_data, voice='bass', is_training=True)

    # Load generated results
    print("\nLoading generated results...")
    try:
        with open("OUTPUT/satb_predictions.json", 'r') as f:
            generated_data = json.load(f)
            print(f"Loaded {len(generated_data)} sequences from predictions")
    except FileNotFoundError:
        print("Error: Could not find predictions file")
        return

    # Calculate generated distributions
    generated_dist_all = calculate_note_distribution(generated_data)
    generated_dist_bass = calculate_note_distribution(generated_data, voice='bass')

    # Plot comparisons
    plot_distribution_comparison(
        training_dist_all,
        generated_dist_all,
        'Note Distribution Comparison (All Voices)',
        'evaluation_results/overall_distribution.png'
    )

    plot_distribution_comparison(
        training_dist_bass,
        generated_dist_bass,
        'Note Distribution Comparison (Bass Voice)',
        'evaluation_results/bass_distribution.png'
    )

    print("\nEvaluation complete. Results saved in evaluation_results directory.")

if __name__ == "__main__":
    main() 
