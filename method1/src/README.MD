# Code Usage and Important Comments

  - If you already had *imutils*  installed, make sure it is at least version v0.3.1, which includes the implementation of the **non_max_suppression**  function, along with a few other minor updates. This method is important for avoiding extra bounding boxes in our output, which helps to reduce *false-positives*. For more information on how Non-Maximum Supression works, refer to [here](https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/).
  
  - Use the following argument format for running the code with images as input:
  
      python ped_detect.py --images path\to\your\directory
   
