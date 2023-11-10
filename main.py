import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from midiutil import MIDIFile
from pydub import AudioSegment


# Define the RNN model
class MusicGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MusicGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last time step's output
        return out


# Function to convert audio file to spectrogram
def audio_to_spectrogram(file_path, input_size, mp3):
    if mp3:
        audio = AudioSegment.from_mp3(file_path)
    else: 
        audio = AudioSegment.from_wav(file_path)
    samples = audio.get_array_of_samples()

    # Normalize samples to the range [0, 1]
    samples = np.array(samples) / np.max(np.abs(samples))

    # Reshape samples into a spectrogram-like format
    spectrogram = samples.reshape(1, -1, input_size)
    return torch.tensor(spectrogram).float()


# Function to generate input and target sequences from audio file
def generate_data_from_audio(file_path, seq_length, input_size, output_size, mp3):
    spectrogram = audio_to_spectrogram(file_path, input_size, mp3)
    spectrogram = torch.tensor(spectrogram)

    # Assuming the spectrogram has sufficient length for the sequence
    input_sequence = spectrogram[:, :seq_length, :input_size]
    target_sequence = spectrogram[:, :seq_length, :output_size]
    
    return input_sequence, target_sequence


# Function to convert 8-bit values to MIDI note numbers
def quantized_to_midi(quantized_values, min_midi_note=30, max_midi_note=90):
    return (quantized_values * (max_midi_note - min_midi_note)).astype(
        int
    ) + min_midi_note


# Directory containing your audio files
audio_directory = "data\\"

# Hyperparameters
input_size = 25  # Input features
hidden_size = 50  # Hidden layer size
output_size = 25  # Output features (increased for more notes)
seq_length = 100000  # Length of input sequence (increased for more notes)
learning_rate = 0.01
num_epochs = 10
quantization_levels = 256  # 8-bit quantization

# Check if a GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize the model and move it to the GPU
model = MusicGenerator(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for file_name in os.listdir(audio_directory):
        if file_name.endswith(".wav"):
            try:
                audio_file_path = "data\\" + file_name
                input_seq, target_seq = generate_data_from_audio(
                    audio_file_path, seq_length, input_size, output_size, False
                )

                # Move input and target sequences to the GPU
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)

                # Forward pass
                output = model(input_seq)

                # Compute loss
                loss = criterion(output, target_seq)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        else:
            try:
                audio_file_path = "data\\" + file_name
                input_seq, target_seq = generate_data_from_audio(
                    audio_file_path, seq_length, input_size, output_size, True
                )

                # Move input and target sequences to the GPU
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)

                # Forward pass
                output = model(input_seq)

                # Compute loss
                loss = criterion(output, target_seq)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Print the average loss for the epoch
    average_loss = total_loss / len(os.listdir(audio_directory))
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}")


# Generate a longer music sequence using the trained model
generated_input = (
    torch.randint(0, quantization_levels, (1, seq_length, input_size)).float()
    / quantization_levels
)
generated_output = model(generated_input)

# Convert quantized values to MIDI note numbers
midi_notes = quantized_to_midi(generated_output.squeeze().detach().numpy())

# Create a MIDI file with multiple tracks
midi = MIDIFile(1)
track = 0
time = 0

# Iterate over the NumPy array
for note in np.nditer(midi_notes):
    midi.addNote(track, 0, note, time, 1, 100)
    time += 1

# Save the MIDI file
with open("generated_music.mid", "wb") as midi_file:
    midi.writeFile(midi_file)
