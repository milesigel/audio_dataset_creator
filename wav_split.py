import os
import csv
import argparse
import pydub.silence
from pydub import AudioSegment
from pydub.playback import play
import sys

# arg parser
parser = argparse.ArgumentParser()
parser.add_argument('-sf', '--sound_file', type=str, required=True, help="path to soundfile")
parser.add_argument('-sd', "--save_dir", type=str, required=False, help="path to save dir")
parser.add_argument('-ss', '--sample_size', type=int, required=False, help="the size of each sample")
parser.add_argument('-cv', '--csv_save', type=bool, required=False, help="True if metadata")
parser.add_argument('-head', '--head', type=str, required=False, help="header to wav files")
parser.add_argument('-class', '--class_id', type=int, required=False, help="classID -- will default to 1")
parser.add_argument('-name', '--dataset_name', type=str, required=False, help="name of the dataset")
args = parser.parse_args()


def slice(sound_file, save_dir, sample_size=10, csv_save=True, head="sample", class_id=1, dataset_name=None):
    sound = AudioSegment.from_file(sound_file, format="wav")
    if not (dataset_name == None):
        folder = os.path.join(save_dir, dataset_name)
    else:
        folder = save_dir

    csv_save_dir = os.path.join(folder, "metadata")
    audio_save_dir = os.path.join(folder, "audio")

    # convert to seconds
    jump = sample_size * 1000
    start = 0
    end = jump
    num_samples = (len(sound) // jump)
    csv_file = []

    print("Creating folders...")

    if not os.path.isdir(folder):
        os.mkdir(folder)
        os.mkdir(audio_save_dir)
        if csv_save:
            os.mkdir(csv_save_dir)
    else:
        if not os.path.isdir(audio_save_dir):
            os.mkdir(audio_save_dir)
            if csv_save:
                os.mkdir(csv_save_dir)

    print("Splitting into samples...")

    for sample in range(num_samples - 1):
        segment = sound[start:end]
        if len(segment) != 10000:
            continue
        file_name = f"{head}_{sample}.wav"
        segment.export(os.path.join(audio_save_dir, file_name), format='wav')
        length = str(len(segment))
        row = [file_name, length, class_id]
        csv_file.append(row)
        start += jump
        end += jump
    if csv_save:
        print("Saving CSV...")
        with open(os.path.join(csv_save_dir, 'meta.csv'), 'w') as file:
            writer = csv.writer(file)
            writer.writerows(csv_file)
    print("Finished.")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    These are the different parameters for saving and choosing how to splice file. WAV files will be saved
    with a RIFF header and can be read in ML applications. 
    """

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

    if len(sys.argv) > 1:
        slice(args.sound_file, args.save_dir,
              sample_size=args.sample_size if args.sample_size is not None else sample_size,
              csv_save=args.csv_save if args.csv_save is not None else csv_save,
              head=args.head if args.head is not None else head,
              class_id=args.class_id if args.class_id is not None else class_id,
              dataset_name=args.dataset_name if args.dataset_name is not None else dataset_name)
        exit()


    slice(sound_file, save_dir, sample_size=sample_size, csv_save=csv_save, head=head, class_id=class_id, dataset_name=dataset_name)
