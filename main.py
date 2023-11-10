import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mido
from midiutil import MIDIFile
from pydub import AudioSegment
import torch.nn.functional as F


# Define the RNN model
class MusicGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MusicGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_note = nn.Linear(hidden_size, output_size)
        self.fc_duration = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        notes = self.fc_note(out[:, -1, :])  # Use the last time step's output for notes
        durations = torch.exp(
            self.fc_duration(out[:, -1, :])
        )  # Exponential for positive durations
        return notes, durations


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


# Function to convert MIDI file to spectrogram-like format
def midi_to_spectrogram(file_path, seq_length, input_size):
    mid = mido.MidiFile(file_path)

    # Extract notes and durations from MIDI file
    notes = []
    durations = []

    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" or msg.type == "note_off":
                notes.append(msg.note)
                durations.append(msg.time)

    # Convert notes and durations to torch tensor
    notes = torch.tensor(notes).float().view(1, -1, 1)
    durations = torch.tensor(durations).float().view(1, -1, 1)

    # Pad or truncate to the desired sequence length
    if notes.shape[1] < seq_length:
        pad_length = seq_length - notes.shape[1]
        notes = F.pad(notes, (0, pad_length, 0, 0))
        durations = F.pad(durations, (0, pad_length, 0, 0))
    elif notes.shape[1] > seq_length:
        notes = notes[:, :seq_length, :]
        durations = durations[:, :seq_length, :]

    # Normalize notes to the range [0, 1]
    notes /= 127.0

    # Stack notes and durations to form the input sequence
    spectrogram = torch.cat((notes, durations), dim=2)
    return spectrogram


# Function to generate input and target sequences from audio or MIDI file
def generate_data(file_path, seq_length, input_size, output_size):
    if file_path.endswith(".mid") or file_path.endswith(".midi"):
        spectrogram = midi_to_spectrogram(file_path, seq_length, input_size)
    else:
        spectrogram = audio_to_spectrogram(file_path, input_size, mp3)

    # Assuming the spectrogram has sufficient length for the sequence
    input_sequence = spectrogram[:, :seq_length, :input_size]
    target_sequence = spectrogram[:, :seq_length, input_size:]

    return (
        input_sequence,
        target_sequence[:, :, :output_size],
        target_sequence[:, :, output_size:],
    )


# Directory containing your audio and MIDI files
audio_directory = "data/"

# Hyperparameters
input_size = 250  # Input features
hidden_size = 500  # Hidden layer size
output_size = 250  # Output features (increased for more notes)
seq_length = 1000  # Length of input sequence (increased for more notes)
learning_rate = 0.01
num_epochs = 10
quantization_levels = 16  # 8-bit quantization

# Check if a GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize the model and move it to the GPU
model = MusicGenerator(input_size, hidden_size, output_size).to(device)
criterion_note = nn.MSELoss()
criterion_duration = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Get a list of audio and MIDI files in the specified directory
audio_data = [
    os.path.join(audio_directory, file_name)
    for file_name in os.listdir(audio_directory)
    if file_name.endswith(".mp3") or file_name.endswith(".wav")
]

midi_data = [
    os.path.join(audio_directory, file_name)
    for file_name in os.listdir(audio_directory)
    if file_name.endswith(".mid") or file_name.endswith(".midi")
]

# Combine audio and MIDI data
data = audio_data + midi_data

# Training loop
for epoch in range(num_epochs):
    total_loss_note = 0
    total_loss_duration = 0
    for file_name in data:
        try:
            input_seq, target_notes, target_durations = generate_data(
                file_name,
                seq_length,
                input_size,
                output_size,
            )

            # Move input and target sequences to the GPU
            input_seq, target_notes, target_durations = (
                input_seq.to(device),
                target_notes.to(device),
                target_durations.to(device),
            )

            # Forward pass
            output_notes, output_durations = model(input_seq)

            # Compute loss for notes and durations
            loss_note = criterion_note(output_notes, target_notes)
            loss_duration = criterion_duration(output_durations, target_durations)

            # Combine the losses (you can adjust the weights for notes and durations)
            loss = loss_note + loss_duration

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_note += loss_note.item()
            total_loss_duration += loss_duration.item()
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Print the average losses for the epoch
    average_loss_note = total_loss_note / len(data)
    average_loss_duration = total_loss_duration / len(data)
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Average Loss (Note): {average_loss_note:.4f}, Average Loss (Duration): {average_loss_duration:.4f}"
    )

# Generate music using the trained model
random_file = np.random.choice(data)
if random_file.endswith(".mp3"):
    mp3 = True
else:
    mp3 = False
generated_input, _, _ = generate_data(
    random_file, seq_length, input_size, output_size, mp3
)
generated_input = generated_input.to(device)
generated_output_notes, generated_output_durations = model(generated_input)

# Convert quantized values to MIDI note numbers
midi_notes = (generated_output_notes.squeeze().detach().numpy() * 127).astype(int)

# Create a MIDI file with a single track and specify instrument
midi = MIDIFile(1)
track = 0
time = 0
instrument = 54  # 8 corresponds to the "Celesta" instrument, which is more bell-like
midi.addProgramChange(track, 0, 0, instrument)

# Adjust the dimension of generated_output_durations
generated_output_durations = generated_output_durations.squeeze().detach().numpy()

for note, duration in zip(midi_notes, generated_output_durations):
    midi.addNote(track, 0, note, time, duration, 100)
    time += duration * 2  # Adjust this value to control the time between notes

# Save the MIDI file
with open("generated_music.mid", "wb") as midi_file:
    midi.writeFile(midi_file)
    print("wrote to file")
