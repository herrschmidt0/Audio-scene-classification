import librosa

class FeatureExtractor:

	def __init__(self, y, sr):
		self.y = y
		self.sr = sr

	def sftf(self):

		stft = librosa.core.stft(self.y, n_fft=1024, hop_length=512)
		stft_log = librosa.amplitude_to_db(abs(stft))
		return stft_log

	def mel(self):
		
		spec = librosa.feature.melspectrogram(self.y, sr=self.sr, n_mels=96, n_fft=2048, hop_length=512)
		return spec

	def mfcc(self):
		...

