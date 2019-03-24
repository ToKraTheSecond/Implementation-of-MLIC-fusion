**NOT FINISHED**

# Implementation-of-MLIC-fusion paper
Multiscale Shape and Detail Enhancement from Multi-light Image Collections; Fattal, Agrawal, Rusinkiewicz

Paper covers fusion of color images set. Each is taken from same camera spot, but with different lightning directions. It is possible to affect output enhancement via two parameters - *alpha* nad *beta* -> perception of depth vs visible detail in the shadow regions

## Paper link
* [princeton.edu](https://gfx.cs.princeton.edu/pubs/Fattal_2007_MSA/mlic.pdf)
* [berkeley.edu](http://kneecap.cs.berkeley.edu/papers/mlic/mlic-SIG07.pdf)


## Dependencies

Paper is implemented in Python 3 with use of NumPy nad OpenCV (only for RGB <-> YUV conversion).
