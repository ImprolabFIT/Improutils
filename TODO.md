Here are various notes and questions about the functions and their functionalities

 - `other.py - order_points()`:
   - is it supposed to work even for more points than 4?
   For example, shall the function restrict 6 points an invalid input?
 - in `\recognition\image_features.py`, if multiple contours are found, then the first
 one only is taken and considered, so for multiple contours the behavior is undefined?
 - `ShapeDescriptors.form_factor` will crash of division by `0`, if `cv2.arcLength()` returns `0`
 - all of the `ShapeDescriptors` functions can crash of division by zero error
