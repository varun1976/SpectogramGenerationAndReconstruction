#!/usr/bin/env python3
import numpy as np
import librosa
import librosa.display
import soundfile
import argparse
import os
import json
from PIL import Image
from PIL import PngImagePlugin
from spsi import spsi

# ------------------------------------------------------
# get any args provided on the command line
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('filename', type=str, help='Name of log mag spectrogram. Include extension')
parser.add_argument('--outdir', type=str, help='Output directory', default='./output')
parser.add_argument('--scalemax', type=int, help='Value to use as the max when scaling from png [0,255] to original [min,max]', default=255)
parser.add_argument('--scalemin', type=int, help='Value to use as the min when scaling from png [0,255] to original [min,max]', default=0)
parser.add_argument('--sr', type=int, help='Samplerate', default=48000)
parser.add_argument('--hopsize', type=int, help='Size of frame hop through sample file', default=256)
parser.add_argument('--glsteps', type=int, help='Number of Griffin&Lim iterations following SPSI', default=50)
parser.add_argument('--wavfile', type=str, help='Optional name for output audio file. Unspecified means use the png filename', default=None)

FLAGS, unparsed = parser.parse_known_args()

def inv_log(img):
	img = np.exp(img) - 1.
	return img

def PNG2LogSpect(fname,scalemin,scalemax):
	"""
	Read png spectrograms, expand to original scale and return numpy array.
	If not stored in one of png metadata, the values needed to undo previous scaling are required to be specified.
	"""
	img = Image.open(fname)
	info = PngImagePlugin.PngInfo()

	try:
		lwinfo = json.loads(img.text['meta'])
	except:
		print('PNG2LogSpect: no img.text, using user specified values!')
		lwinfo = {}
		lwinfo['scaleMin'] = scalemin #require to pass in
		lwinfo['scaleMax'] = scalemax
		info.add_text('meta',json.dumps(lwinfo))

	minx, maxx = float(lwinfo['scaleMin']), float(lwinfo['scaleMax'])

	img = img.convert('L')
	outimg = np.asarray(img, dtype=np.float32)
	outimg = (outimg - 0)/(255-0)*(maxx-minx) + minx

	return np.flipud(outimg), lwinfo

D,_ = PNG2LogSpect(FLAGS.filename,FLAGS.scalemin,FLAGS.scalemax)
Dsize, _ = D.shape
fftsize = 2*(Dsize-1) #infer fftsize from no. of fft bins i.e. height of image

magD = inv_log(D)
y_out = spsi(magD, fftsize=fftsize, hop_length=FLAGS.hopsize)

if FLAGS.glsteps != 0 : #use spsi result for initial phase
	x = librosa.stft(y_out, n_fft=fftsize, hop_length=FLAGS.hopsize, center=False)
	p = np.angle(x)
	for i in range(FLAGS.glsteps):
		S = magD * np.exp(1j*p)
		y_out = librosa.istft(S, hop_length=FLAGS.hopsize, center=True) # Griffin Lim, assumes hann window, librosa only does one iteration?
		p = np.angle(librosa.stft(y_out, n_fft=fftsize, hop_length=FLAGS.hopsize, center=True))

scalefactor = np.amax(np.abs(y_out))
print('Scaling peak sample, ' + str(scalefactor) + ' to 1')

if FLAGS.wavfile == None:
	soundfile.write(os.path.splitext(FLAGS.filename)[0]+'.wav', y_out, FLAGS.sr)
else:
	soundfile.write(FLAGS.wavfile, y_out, FLAGS.sr)
