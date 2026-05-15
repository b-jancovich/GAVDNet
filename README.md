# GAVDNet

## Intro

GAVDNet (Generalized Animal Vocalisation Detector Network) is a MATLAB-based system for the automated detection of stereotyped animal vocalizations in passive acoustic monitoring (PAM) recordings.

GAVDNet’s audio augmentation pipeline takes a small number of clean example recordings of a target call and uses a physically informed data augmentation pipeline to generate a large, diverse training dataset, injecting real world background noise into the final training samples at various SNRs. The synthetic dataset is used to fine-tune a pretrained neural network to recognise the target call.

The system is designed exclusively for stereotyped vocalizations: calls that have a consistent, recognisable time-frequency structure across individuals and over time, though mechanisms are included to handle some inter-individual and inter-annual variability in call frequency. GAVDNet was developed and validated for blue whale songs. It may work well for other stereotyped sounds, though this remains untested. The data augmentation system applies audio processing that mimics acoustic phenomena like Doppler shift, reverberation, and transmission loss.

Training a single detector model typically takes 1 to 8 hours depending on your hardware, training hyperparameters and the number of training sequences. Inference runs on the trained detector are typically processed at ~150x real-time audio playback speed. 4 hours of 32-bit audio sampled at 250Hz Fs should take approximately 90 seconds, including pre and post processing.

## Software requirements:

• MATLAB R2024a or later  
• Audio Toolbox  
• Deep Learning Toolbox  
• Parallel Computing Toolbox (recommended, not strictly required)  
• Signal Processing Toolbox  
• Wavelet Toolbox (for data preparation utilities)  
• The customAudioAugmenter repository: https://github.com/b-jancovich/customAudioAugmenter  

## Hardware recommendations:

• 64 GB RAM minimum  
• NVIDIA GPU with 8 GB or more memory and CUDA support  
• ~500 GB or more free disk space, preferably on an SSD  
• Multi-core CPU  

## See GAVDNet\_User\_Guide.pdf for full instructions on installation and use
