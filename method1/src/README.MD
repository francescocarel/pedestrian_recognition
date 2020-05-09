# Code Usage and Important Comments

  - If you already had *imutils*  installed, make sure it is at least version v0.3.1, which includes the implementation of the **non_max_suppression**  function, along with a few other minor updates. This method is important for avoiding extra bounding boxes in our output, which helps to reduce *false-positives*. For more information on how Non-Maximum Supression works, refer to [here](https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/).
  
  - Use the following argument format for running the code with images as input:
  
        python ped_detect.py --images path\to\your\directory
   
  - The HOG method requires the *detectMultiScale* parameters to be set properly depending on your the images that you're processing and to tune the trade-off between accuracy and performance (especially important for real-time video processing).
 
 ### The Parameters to the cv2.HOGDescriptor(). DetectMultiScale Function:
 
    - img (required)
    - hitThreshold (optional)
    - winStride (optional)
    - padding (optional)
    - scale (optional)
    - finalThreshold (optional)
    - useMeanShiftGrouping (optional)
    
The values of these parameters are of extreme importance, as they can:

  - Increase the number of false-positive detections (i.e., reporting that a location in an image contains a person, but when in reality it does not).
  
  - Result in missing a detection entirely.

  - Dramatically affect the speed of the detection process
