import click
import os
import subprocess
import soundfile as sf
import math
import numpy
import json


def convert_to_wav(input_filename, output_filename, samplerate=44100, nbits=16, nchannels=1):
    """
    Converts any audio file type to pcm audio with samplerate and nbits as specified in parameters.
    Conversion uses ffmpeg therefore this must be installed in the system.
    :param input_filename: input file path
    :param output_filename: output file path
    :param samplerate: samplerate of output file
    :param nbits: bitdepth of output file
    """
    nbits_labels = {8: 'pcm_u8', 16: 'pcm_s16le', 24: 'pcm_s24le', 32: 'pcm_s32le'}
    cmd = ["ffmpeg", "-y", "-i", input_filename, "-acodec", nbits_labels[nbits],
           "-ar", str(samplerate), "-map_metadata", "-1"]
    if nchannels == 0:
        cmd += ["-ac", str(nchannels)]
    elif nchannels == 1:
        cmd += ["-map_channel", "0.0.0"]
    cmd += [output_filename]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = process.communicate()
    if process.returncode != 0 or not os.path.exists(output_filename):
        raise Exception("Failed converting to wav data:\n" + " ".join(cmd) + "\n" + stderr + "\n" + stdout)

def post_process_file(input_filename, output_filename):
    signal, samplerate = sf.read(input_filename)
    n_samples = len(signal)
    n_samples_min_unit = samplerate * 0.02  # 20 ms
    n_samples_new_signal = int(int(math.ceil(n_samples/n_samples_min_unit)) * n_samples_min_unit)
    new_signal = numpy.random.normal(loc=0, scale=1.0/32768.0, size=n_samples_new_signal)
    new_signal[0: n_samples] = new_signal[0: n_samples] + signal
    sf.write(output_filename, new_signal, samplerate)
    return float(n_samples)/samplerate  # return duration


def convert_audio_files_to_wav(dir, outdir, sr, nb, nc, suffix, max):
    if outdir == '':
        outdir = dir

    if nb not in [8, 16, 24, 32]:
        click.echo("Setting number of bits to 16 as provided number is not supported")
        nb = 16

    try:
        durations = json.load(open('durations.json'))
    except IOError:
        durations = {}

    with click.progressbar([fname for fname in os.listdir(dir)[:max]
                            if not fname.startswith('.')],
                           label="Converting audio files...") as dir_filenames:
        for filename in dir_filenames:
            filename, extension = os.path.splitext(filename)
            input_filename = os.path.join(dir, "%s%s" % (filename, extension))
            output_filename = os.path.join(outdir, "%s%s.wav" % (filename, suffix))
            if not os.path.exists(output_filename):  # don't process if already done
                try:
                    convert_to_wav(input_filename, output_filename, samplerate=sr, nbits=nb, nchannels=nc)
                    duration = post_process_file(output_filename, output_filename)
                    durations[filename.split('.')[0]] = duration
                except Exception, e:
                    click.echo(e)

    json.dump(durations, open('durations.json', 'w'))


@click.command()
@click.argument('dir')
@click.option('--outdir', default='', help='Output directory to save the files (by default, same as input).')
@click.option('--sr', default=44100, help='Samplerate of the output files (default 44100).')
@click.option('--nb', default=16, help='Bitdepth of the output files (default 16).')
@click.option('--nc', default=0, help='Number of channels of the output files (default 0, keeps same number of channels).')
@click.option('--suffix', default='', help='Suffix to be appended to output filenames (default '').')
@click.option('--max', default=999999999, help='Max number of sounds to process.')
def main(dir, outdir, sr, nb, nc, suffix, max):
    """
    Program to convert to wav all audio files present in the directory DIR.
    Output files will be saved in the same input directory with a SUFFIX appended to the filenames to distinguish
    from original files in case the original file extension was already wav.
    Output samplerate and bitdepth can be specified using the SR and NBITS options.
    """
    convert_audio_files_to_wav(dir, outdir, sr, nb, nc, suffix, max)

if __name__ == '__main__':
    main()
