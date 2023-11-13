import os
from pydub import AudioSegment
from midi2audio import FluidSynth

def convert_midi_to_wav(midi_file, output_folder):
    fs = FluidSynth()
    wav_file = os.path.join(output_folder, os.path.splitext(os.path.basename(midi_file))[0] + ".wav")
    fs.midi_to_audio(midi_file, wav_file)
    return wav_file

def batch_convert_midi_to_wav(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".mid") or file_name.endswith(".midi"):
            midi_file = os.path.join(input_folder, file_name)
            convert_midi_to_wav(midi_file, output_folder)
            print(f"Converted: {midi_file}")

if __name__ == "__main__":
    input_folder = "data/"
    output_folder = "data/"

    batch_convert_midi_to_wav(input_folder, output_folder)
