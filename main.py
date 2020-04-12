import wave
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import threading
from scipy.fftpack import fft

CHUNK = 1024 * 4
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
WAVE_OUTPUT_FILENAME = "voice.wav"

callback_output = []
audio_data = []
sampling_active = True


def key_capture_thread():
    global sampling_active
    input()
    sampling_active = False


def callback(in_data, frame_count, time_info, flag):
    global audio_data
    audio_data = np.frombuffer(in_data, dtype=np.int16)
    callback_output.append(audio_data)
    return audio_data, pyaudio.paContinue


def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1,
                    rate=RATE, input=True,
                    output=False,
                    frames_per_buffer=CHUNK,
                    stream_callback=callback)

    key_press_thread = threading.Thread(target=key_capture_thread,
                                        args=(),
                                        name='key_capture_thread',
                                        daemon=True)

    plt.ion()
    fig, (ax, ax2) = plt.subplots(2)
    ax.grid()
    ax2.grid()
    x = np.arange(0, CHUNK)
    x_fft = np.linspace(0, RATE, CHUNK)
    line, = ax.plot(x, np.random.rand(CHUNK))
    line_fft, = ax2.semilogx(x_fft, np.random.rand(CHUNK))
    ax.set_ylim(-2**15, 2**15)
    ax.set_xlim(0, CHUNK)
    ax2.set_xlim(20, RATE/2)

    stream.start_stream()
    key_press_thread.start()

    while stream.is_active() and sampling_active == True:
        line.set_ydata(audio_data)
        y_fft = fft(audio_data)
        line_fft.set_ydata(np.abs(y_fft[0:CHUNK]*2/(256 * CHUNK)))
        fig.canvas.draw()
        fig.canvas.flush_events()

    stream.stop_stream()
    stream.close()
    p.terminate()
    plt.close()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(callback_output))
    wf.close()
    print('ending.....')


if __name__ == "__main__":
    main()

