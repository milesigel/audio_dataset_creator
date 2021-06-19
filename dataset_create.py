import pydub
import torch
import torchaudio
import os
import csv
import argparse
import sys
from wav_split import slice
import pandas as pd
from torch.utils.data import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('-sf', '--sound_file', type=str, required=False, help="path to soundfile")
parser.add_argument('-sd', "--save_dir", type=str, required=False, help="path to save dir")
parser.add_argument('-ss', '--sample_size', type=int, required=False, help="the size of each sample")
parser.add_argument('-cv', '--csv_save', type=bool, required=False, help="True if metadata")
parser.add_argument('-head', '--head', type=str, required=False, help="header to wav files")
parser.add_argument('-class', '--class_id', type=int, required=False, help="classID -- will default to 1")
parser.add_argument('-name', '--dataset_name', type=str, required=False, help="name of the dataset")
args = parser.parse_args()


# case1 : have an audio file that is to long and want to spit it.


class Audio_Dataset_Creator():

    def __init__(self, sound_path, data_path, csv_save=True, class_id=1, dataset_name=None):
        self.sound_file = sound_path
        self.data_path = data_path
        self.sample_size = None
        self.csv_save = csv_save
        self.label = class_id
        self.dataset_name = dataset_name

    def split_and_save(self, sample_size=10, head="sample"):
        slice(self.sound_file, self.data_path, sample_size=sample_size, csv_save=self.csv_save, head=head,
              class_id=self.label,
              dataset_name=self.dataset_name)


class Audio_Dataset(Dataset):

    def __init__(self, audio_dir, csv_dir, transform, target_sample_rate, resample_rate=16000):
        self.audio_dir = audio_dir
        self.csv_dir = pd.read_csv(csv_dir)
        self.transform = transform
        self.target_sample_rate = target_sample_rate

    def __getitem__(self, index):
        path_to_audio = self._get_audio_path(index)
        label = self._get_label(index)
        signal, sr = torchaudio.load(path_to_audio)
        signal = self._resample(signal, sr)
        signal = self._mix_down(signal)
        signal = self.transformation(signal)
        return signal, label

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _resample(self, signal, sr):
        if sr is not self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
            return signal
        else:
            return signal

    def _get_label(self, index):
        """
        will need to get replaced when changing csv locations
        :param index:
        :return: label of single file
        """
        label = self.csv_dir.iloc[index, 2]
        return label

    def _get_audio_path(self, index, inner=None):
        ending = self.csv_dir.iloc(index, 0)
        if inner is not None:
            path = os.path.join(self.audio_dir, inner, ending)
        else:
            path = os.path.join(self.audio_dir, ending)
        return path

    def __len__(self):
        return len(self.csv_dir)


if __name__ == "__main__":
    """
       These are the different parameters for saving and choosing how to splice file. WAV files will be saved
       with a RIFF header and can be read in ML applications. 
       """

    DATASET_MODE = True
    AUDIO_MODIFY = False

    if DATASET_MODE == True:
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=2048,
            hop_length=512,
            n_mels=64
        )
        SAMPLE_RATE = 16000
        new_set = Audio_Dataset()
    ### For making a dataset object

    # name of the data set
    dataset_name = "sad"
    # path to original sound file
    sound_file = "/Users/milessigel/Desktop/Datasets/piano_sentiment_dataset/sad_music.wav"
    # where are the top level datasets saved
    save_dir = "/Users/milessigel/Desktop/Datasets"
    # class_id
    class_id = 1
    # header for the sample files
    head = "sample"
    # do you want CSV?
    csv_save = True
    # number of seconds of sample
    sample_size = 10
    if (AUDIO_MODIFY):
        if len(sys.argv) > 1:
            working = Audio_Dataset_Creator(args.sound_file, args.save_dir,
                  csv_save=args.csv_save if args.csv_save is not None else csv_save,
                  class_id=args.class_id if args.class_id is not 1 else class_id,
                  dataset_name=args.dataset_name if args.dataset_name is not None else dataset_name)
            slice(sample_size=args.sample_size if args.sample_size is not 10 else sample_size,
                  head=args.head if args.head is not None else head)
            exit()
        else:
            working = Audio_Dataset_Creator(sound_file,
                                        save_dir,
                                        csv_save=csv_save,
                                        class_id=class_id,
                                        dataset_name=dataset_name)

            slice(sample_size=sample_size,
              head=head)
            exit()


