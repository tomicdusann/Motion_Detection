# Motion_Detection
Detection of moving objects from a video source, developed in two versions - CPU multithreaded C++ implementation and GPU/CUDA C/C++ implementation. Input video source its being processed in a way for program pipeline to convert it from RGB to grayscale in order generate binary mask of movement, make it more stable and clear and also draw detection boxes around moving objects on RGB frame. Basically we have raw color frame as input and binary mask frame and color frame with added detection boxes as an output. 

Link to provided playlist on YT with two demo videos: https://www.youtube.com/watch?v=VXUxS-2WmoM&list=PLqgdQ5oU0ynbW0gKUi74BRAftfn82XyqZ
