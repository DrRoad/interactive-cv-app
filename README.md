# Interactive CV App

A desktop (and soon to be phone) app which applies traditional computer vision algorithms to the device's camera stream. The user has the ability to change the different parameters of the underlying CV algorithms, so you can see their effects in real time.

![demo](media/demo.gif)

The app was created using [Kivy](https://github.com/kivy/kivy) for GUI development, and [OpenCV](https://github.com/opencv/opencv) for the computer vision algorithms.

This app is fully functioning on desktop. I'm currently developing a mobile version of the app (android, then iOS).

## To run this app

1. Download and extract the repository to your computer
2. In the terminal, go to the directory containing main.py
3. Install necessary dependencies (see below)
4. Run the command "python main.py"
5. A window should pop up containing the app. Make sure you have a working webcam, and the correct permissions enabled for the app to access it.

By default, the app is scaled to replicate the size of a phone screen (for debug purposes), but you can adjust the size of the window if you'd prefer.

### Dependencies

This app uses OpenCV and Kivy, which can be installed via the following pip commands:

> pip install opencv-python
>
> pip install kivy

This app was built and tested in python 3.6, but I think it should work for all python 3+ versions.

## Computer Vision Algorithms

The app allows you to play with the following traditional CV Algorithms:

1. Canny Edge Detection - Allows you to toggle the two threshold values.
2. Threshold (Binary) - Can set a fixed threshold value, or alternatively use Otsu's algorithm to find the best threshold.
3. Threshold (Adaptive) - Can vary the size of the pixel neighborhood considered, the kernel type (Gaussian or box), and the mean offset that's subtracted from each pixel.
4. Blur - A simple Gaussian blur, can change the kernel size. May add more options later, such as varying the stdev, or changing the kernel type.
5. Optical Flow - A dense optical flow calculation, based on the Gunnar-Farneback algorithm. Color corresponds to the direction of local flow, intensity corresponds to speed.
6. Convolution - A variety of widely used kernel types (Sobel, sharpen, outline, etc) for convolving the image. Can vary the kernel size.

## Future Releases

Currently working on the following:

1. Port app to android and ultimately Google Play Store
2. Port to iOS and Apple Store

## About me

I'm a Physics PhD with 6+ years of experience working with big data, now pursuing a career in computer vision and deep learning. If you have any questions about me, this project, or something else, feel free to reach out:

* [linkedin](https://www.linkedin.com/in/jeffsrobertson/)

