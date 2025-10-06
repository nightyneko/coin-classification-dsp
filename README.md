# coin-classification-dsp

### overview

we implement coin classification for 1 5 and 10 bath by should charecteristic such as spectral flatness, spectral bandwidth and spectral centroid

### method 
- features extraction

Sound -> HPF -> HPSS -> spectral centroid,flatness,etc 

- z score for training data

Sound -> extraction -> mean,sd per features -> z score per coin type

- classification

Sound -> extraction -> z-score -> distance d0,d1,d2 -> show coin prediction


