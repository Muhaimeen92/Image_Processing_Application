# Assignment Completion

I have completed all the requirements for this assignment as per the instructions in the Engineering Challenge. 

All the methods and routes in server.py have been completed.

Upon testing the entire application is functioning as expected.

One part of my normalizing algorithm that is not 100% accurate is the cropping_for_normalizing method. This method 
crops the image to the highest multiple of 32 in order to create an exact number of 32 x 32 patches for the 
normalization. This leaves out some of the pixels around the edges. This  method is still quite accurate as majority of
the pixels are being accounted for. 

Given more time, I would be able modify the algorithm to create smaller patches for the edges, compute their means 
and standard deviations to be then able to then calculate the median of all patches of the image.