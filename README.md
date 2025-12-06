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


Link to Dataset: [HERE](https://universe.roboflow.com/contrails-mqdsk/contrails-segmentation-pybpt)


# Results:

# YOLO11l-seg (fused)
**Model summary**
- **Model:** YOLO11l-seg (fused)  
- **Layers:** 203  
- **Parameters:** 27,590,760
- **Gradients:** 0  
- **GFLOPs:** 131.9

### Validation metrics (L)
| Class             | Images | Instances | Box(P) | Box(R) | Box mAP50 | Box mAP50-95 | Mask(P) | Mask(R) | Mask mAP50 | Mask mAP50-95 |
|-------------------|--------|-----------|--------|--------|-----------|--------------|---------|---------|------------|---------------|
| all               | 320    | 856       | 0.636  | 0.489  | 0.549     | 0.346        | 0.626   | 0.479   | 0.526      | 0.247         |
| contrail maybe    | 16     | 20        | 0      | 0      | 0.0633    | 0.0202       | 0       | 0       | 0.0516     | 0.00944       |
| contrail old      | 73     | 99        | 0.55   | 0.47   | 0.479     | 0.314        | 0.49    | 0.417   | 0.391      | 0.15          |
| contrail veryold  | 110    | 245       | 0.669  | 0.661  | 0.708     | 0.497        | 0.656   | 0.649   | 0.676      | 0.313         |
| contrail young    | 124    | 168       | 0.683  | 0.554  | 0.604     | 0.358        | 0.693   | 0.56    | 0.626      | 0.263         |
| parasite          | 90     | 147       | 0.857  | 0.695  | 0.822     | 0.49         | 0.841   | 0.681   | 0.773      | 0.342         |
| sun               | 168    | 168       | 0.925  | 0.821  | 0.932     | 0.533        | 0.932   | 0.827   | 0.93       | 0.54          |
| unknow            | 9      | 9         | 0.768  | 0.222  | 0.234     | 0.208        | 0.769   | 0.222   | 0.236      | 0.112         |



# YOLO11x-seg (fused)

**Model summary**

- **Model:** YOLO11x-seg (fused)  
- **Layers:** 203  
- **Parameters:** 62,011,368  
- **Gradients:** 0  
- **GFLOPs:** 295.9  

---

### Validation metrics (K)

| Class            | Images | Instances | Box P  | Box R  | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|------------------|--------|-----------|--------|--------|-----------|--------------|--------|--------|------------|---------------|
| all              | 320    | 838       | 0.748  | 0.531  | 0.547     | 0.306        | 0.748  | 0.506  | 0.526      | 0.237         |
| contrail maybe   | 14     | 16        | 1      | 0      | 0.107     | 0.0259       | 1      | 0      | 0.0904     | 0.0321        |
| contrail old     | 64     | 88        | 0.465  | 0.398  | 0.385     | 0.198        | 0.441  | 0.352  | 0.318      | 0.0956        |
| contrail veryold | 113    | 250       | 0.657  | 0.616  | 0.654     | 0.378        | 0.667  | 0.6    | 0.613      | 0.257         |
| contrail young   | 127    | 160       | 0.553  | 0.613  | 0.472     | 0.247        | 0.553  | 0.569  | 0.482      | 0.173         |
| parasite         | 82     | 141       | 0.784  | 0.809  | 0.824     | 0.494        | 0.765  | 0.74   | 0.781      | 0.365         |
| sun              | 169    | 169       | 0.883  | 0.853  | 0.901     | 0.486        | 0.9    | 0.853  | 0.912      | 0.51          |
| unknow           | 14     | 14        | 0.895  | 0.429  | 0.484     | 0.315        | 0.907  | 0.429  | 0.488      | 0.229         |

---

**Speed**

- Preprocess: 0.2 ms / image  
- Inference: 4.8 ms / image  
- Postprocess: 1.5 ms / image  
