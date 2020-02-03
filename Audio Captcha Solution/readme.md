# Audio Captcha Solver Using Deep Learning

![TCD Project](https://imgur.com/Y8rbrU3)

The captcha solver is created from TTS audio compiled from various text to speech API available such as Espeak from linux, SAPI from microsoft, watson from IBM etc. in multiple voices such as male, female etc.

The repository contains
  - Different voices mp3 for future research
  - Base python code for training the model
  - The code for classification of the audios
  - All the other contents are mainly for the functioning of the code and some instruction which we found out were better running from the console in parallel to surpass jupiter's memory constraints (for a more robust solution)

# Methodology

  - Created multiple audios for general symbols 0-9 digits and A-Z letters from multiple TTS APIs
  - Combined randomly to generate 8 character captchas
  - With new file we run our code to separate files into individual character again (done to simulate the characters we will receive when we split the out of sample captchas)
  - Trained the model on individual character. Note how we get different types of the same character due to being cut differntly on the basis of silences indentified by pydub
  - Converted audios to mel-spectograms and trained a CNN model over the spectograms
  - Model is ready for classification

### Tech

* [jiaaro/pydub] - https://github.com/jiaaro/pydub
* [librosa/librosa] - https://github.com/librosa/librosa
* [tensorflow/tensorflow] - https://github.com/tensorflow/tensorflow
* [numpy/numpy] - https://github.com/numpy/numpy


Install the dependencies

```sh
$ pip install numpy
$ pip install pydub
$ pip install tensorflow
$ pip install librosa
```