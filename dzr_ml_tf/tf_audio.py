import tensorflow as tf


def to_f32(tensor_tf): # cast to float32
    return tf.cast(tensor_tf, tf.float32)

def round_i64(tensor_tf): # round and cast to int64
    return tf.cast(tf.round(tensor_tf), tf.int64)



def load_audio_waveform(filename_tf, format="mp3", fs=44100, channel_count=2):
    audio_binary = tf.read_file(filename_tf)
    return tf.contrib.ffmpeg.decode_audio(audio_binary, file_format=format, samples_per_second=fs, channel_count=channel_count)

def crop_audio(waveform_tf, seg_start, seg_duration, fs=44100):
    return waveform_tf[round_i64(to_f32(seg_start)*fs):round_i64((to_f32(seg_start)+to_f32(seg_duration))*fs),:]


def stft_tf(waveform_tf, frame_length=1024, frame_step=512, fft_length=1024):
    """
        Compute stft for a multi channel waveform

        Params
        - waveform: TxC tensor
        - frame_length, frame_step, fft_length as defined in tf.contrib.signal.stft

        Return
        - A CxNxF tensor. N: number of time frames, F: number of frequency bins.
    """

    def one_channel_stft(channel, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length):
        return tf.squeeze(tf.contrib.signal.stft(tf.reshape(channel, (1, -1)), frame_length=frame_length, frame_step=frame_step, fft_length=fft_length))

    res_stft = tf.map_fn(one_channel_stft, tf.transpose(waveform_tf), dtype=tf.complex64)
    res_stft.set_shape((None, None, fft_length/2 + 1))
    return res_stft


def mel_spectrogram_tf(waveform, sample_rate=44100., frame_length=1024, frame_step=512, fft_length=1024, lower_edge_hertz=60.0, upper_edge_hertz=8000.0, num_mel_bins=64):
    """
        Compute Mel spectrogram from waveform

        Params
        - waveform: TxC tensor
        - frame_length, frame_step, fft_length as defined in tf.contrib.signal.stft
        - lower_edge_hertz, upper_edge_hertz, num_mel_bins as defined in tf.contrib.signal.linear_to_mel_weight_matrix

        Return
        - A CxNxnum_mel_bins tensor. N: number of time frames.

    """
    magnitude_spectrogram = tf.abs(stft_tf(waveform, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length))
    num_spectrogram_bins = magnitude_spectrogram.shape[-1].value

    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
    mel_spectrogram = tf.tensordot(magnitude_spectrogram, linear_to_mel_weight_matrix, 1)

    mel_spectrogram.set_shape(magnitude_spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    return mel_spectrogram

def mfcc_tf(waveform, n_mfccs=13, frame_length=1024, frame_step=512, fft_length=1024, lower_edge_hertz=60.0, upper_edge_hertz=8000.0, num_mel_bins=64, log_offset=1e-6):
    """
        Compute MFCC from waveform

        Params
        - waveform: TxC tensor
        - n_mfccs: number of mfcc coefficients
        - log_offset: log offset for avoiding null values when taking log of mel spectrogram
        - frame_length, frame_step, fft_length as defined in tf.contrib.signal.stft
        - lower_edge_hertz, upper_edge_hertz, num_mel_bins as defined in tf.contrib.signal.linear_to_mel_weight_matrix

        Return
        - A CxNxnum_mel_bins tensor. N: number of time frames.

    """
    mel_spectrogram = mel_spectrogram_tf(waveform)
    log_mel_spectrogram = tf.log(mel_spectrogram + log_offset)

    # Keep the first `num_mfccs` MFCCs.
    mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :n_mfccs]

    return mfccs



def amplitude_to_db(tensor, amin=1e-13, top_db=120., ref_value=1.):
    """
        Convert amplitude value to decibel
    """
    res = 20.*tf.log(tf.maximum(tensor, amin))/tf.log(10.) - 20.* (tf.log(ref_value)/tf.log(10.))
    return tf.maximum(res, tf.reduce_max(res) - top_db)



if __name__=="__main__":
    waveform = load_audio_waveform(tf.constant("/Users/rhennequin/Downloads/money.mp3"))
    cropped_wf = crop_audio(waveform, tf.constant(10), tf.constant(10))
    signal_mfcc = mfcc_tf(cropped_wf)

    with tf.Session() as sess:
        res = sess.run(signal_mfcc)
