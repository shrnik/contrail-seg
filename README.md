We are trying to detect which airplane in a camera’s view is creating contrails (condensation trails) . We have ADS-B (flights gps locations) data around the camera for a time range. We also have the Intrinsics and extrinsic calibration parameters of the camera. Using these we can project 3D world coordinates into the 2D image plane. 

I have already created a pipeline that uses computer vision techniques like canny edge detection and Hough transform to detect Contrails in the ROI (yellow area behind an airplane). But this is very sensitive to camera calibration and parameters of Canny edge detection and Hough transform.

The second challenge is to know the age of a contrail (is it persistent or not)

For this I am going to use a YOLOv11 model for image segmentation and use the data described in [Paper](https://www.sciencedirect.com/science/article/pii/S2352340925000964)

This dataset has the following classes:

| Class of object | Description |
| --- | --- |
| Maybe contrail | An aircraft trajectory is present. The object looks like a young contrail but has very low contrast. |
| Young contrail | An aircraft trajectory is present. A contrail object can be unambiguously matched to the trajectory. The contrail is less than 3 minutes old. |
| Old contrail | The aircraft trajectory has left the image. The contrail object is still linear but is more than 3 minutes old. |
| Very old contrail | The aircraft is well outside the image. The contrail object is no longer linear and may be heterogeneous. The contrail is likely to be more than 10 to 15 minutes old. |
| Parasite | The object is a feature that may look like an old contrail but that does not move with the wind when looking at successive images. |
| Sun | The object corresponds to the pixels saturated by the Sun but is limited to a disc and does not include sunrays. A Sun object hidden by a cloud is not annotated. |
| Unknown | The feature is a cloud streak that may look like an old contrail or a very old contrail but we cannot determine whether it is a contrail or a natural cirrus. |
| Background | The black pixels in the four corners of the image. |
| Sky | All other pixels in the image. |


Link to Dataset: [HERE](https://universe.roboflow.com/contrails-mqdsk/my-first-project-tpdbj)


I am going to test this on a camera that I have and probably try to make a demo video out of the detections