<div align="center">
<h1>Multi-Camera Calibration</h1>

[![Python Version][badge-python]][link-repo]
[![License][badge-license]](LICENSE)
[![Contribute][badge-contribute]][link-repo]
</div>

The multi-camera calibration module is designed to calibrate multiple cameras and generate the transformation matrices required to create a top-down view of a scene. By leveraging OpenCV and ChArUco boards, this module ensures precise stitching of images and enables accurate distance measurements.

Originally developed for the Hanze team’s participation in the [Self Driving Challenge 2024][link-sdc], this project has since been generalized for broader applications.

## Table of Contents
- [How does it work?](#how-does-it-work)
  - [Step 1. Capture the Images](#step-1-capture-the-images)
  - [Step 2. Find the Corners](#step-2-find-the-corners)
  - [Step 3. Get the Stitched Image](#step-3-get-the-stitched-image)
    - [Step 3.1. Find the Homography](#step-31-find-the-homography)
    - [Step 3.2. Find the Offsets](#step-32-find-the-offsets)
    - [Step 3.3. Crop the Bottom](#step-33-crop-the-bottom)
    - [Step 3.4. Find the Vanishing Point](#step-34-find-the-vanishing-point)
  - [Step 4. Get the Top-Down Image](#step-4-get-the-top-down-image)
    - [Step 4.1. Find the Homography](#step-41-find-the-homography)
    - [Step 4.2. Correct the Angle (Optional)](#step-42-correct-the-angle-optional)
    - [Step 4.3. Calculate the Region of Interest](#step-43-calculate-the-region-of-interest)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)


## How does it work?
### Step 1. Capture the Images
To calibrate the cameras, we need to capture images of the ChArUco board from all cameras. It is extremely important that the ChArUco board is clearly visible in all images. A ChArUco board is a chessboard with ArUco markers on it. The ArUco markers allow us to tell which corner is which, making it extremely useful for stitching images together.

```python
# Import OpenCV
import cv2

# Initialize the cameras.
cam_left = cv2.VideoCapture(0)
cam_center = cv2.VideoCapture(1)
cam_right = cv2.VideoCapture(2)

# Get the frames of each camera.
ret, frame_left = cam_left.read()
ret, frame_center = cam_center.read()
ret, frame_right = cam_right.read()
```

![Base-min](https://github.com/HeadTriXz/SDC-2024/assets/32986761/4cbb5916-dd5a-454c-8d6d-fffe5fb6b939)

---

### Step 2. Find the Corners

Next up is locating the corners of the ChArUco board in all images. OpenCV has a built-in module for dealing with ArUco markers, called `cv2.aruco`. We can use this module to find the corners of the ChArUco board in each image.

![ChArUco-min](https://github.com/HeadTriXz/SDC-2024/assets/32986761/ccc4e0fd-297c-4307-9916-f4ccdbd6470f)

---

### Step 3. Get the Stitched Image
#### Step 3.1. Find the Homography

The next step is to find the homography between the center camera and the other cameras. A homography is a transformation matrix that maps points from one image to another. We can use the homography to make the other cameras have the same perspective as the center camera. This is used to make one big image from all the cameras, giving us more details of the road.

![Homography-min](https://github.com/HeadTriXz/SDC-2024/assets/32986761/6ac23afa-3f42-4587-a3f0-f268a3814e3d)

#### Step 3.2. Find the Offsets

To stitch the images together, we need to find the offsets between the leftmost camera and the other cameras. This is required to ensure that when we combine them, they blend seamlessly and look like one big image.

![Stitched-min](https://github.com/HeadTriXz/SDC-2024/assets/32986761/50b28057-8fbc-4ca7-80e8-7a4c7e827ee6)

#### Step 3.3. Crop the Bottom

The bottom part of the stitched image contains the kart and parts of the road that are not relevant. To declutter the image and speed up processing, we can safely crop the bottom part of the image.

![Cropped Stitched-min](https://github.com/HeadTriXz/SDC-2024/assets/32986761/32d0b388-b3ab-40ee-bbb6-a930fcbe6c97)

#### Step 3.4. Find the Vanishing Point

In order to avoid the Pac-Man Effect, we need to find the vanishing point of the road and crop away everything above it. We can do this by finding the intersection of the ChArUco board's lines.

> [!NOTE]
> **Pac-Man Effect**: The belief that someone attempting to go over the edge of the flat Earth would teleport to the other side.

> Table 1: On the left is an image illustrating the calculation of the vanishing point, while on the right is the outcome after cropping.

| Before Cropping                                                                                                            | After Cropping                                                                                                                |
|----------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| ![Stitched Vanishing Line-min](https://github.com/HeadTriXz/SDC-2024/assets/32986761/df6165ed-9fe4-43e8-bab2-5cc0282736bd) | ![Stitched Vanishing Cropped-min](https://github.com/HeadTriXz/SDC-2024/assets/32986761/cc10a37f-15d9-4140-8f8d-ca6158cc7e39) |

---

### Step 4. Get the Top-Down Image
#### Step 4.1. Find the Homography

To get a top-down view of the road, we need to find the homography between the real-world grid of the ChArUco board and the ChArUco board that we see in the image. A top-down view is useful for path planning, calculating distances, and the curvature of the road.

> Table 2: On the left is an image showing the homography between ChArUco board points in the image and the real world. On the right is the resulting top-down view.

| Before Transform                                                                                                      | After Transform                                                                                                         |
|-----------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| ![Topdown-Homography-min](https://github.com/HeadTriXz/SDC-2024/assets/32986761/06cbad20-7c8c-49af-98df-b3bbd9275679) | ![Angle-Corrected Top-Down](https://github.com/HeadTriXz/SDC-2024/assets/32986761/3ff2942b-cc24-4709-83f0-996542ef47e4) |

#### Step 4.2. Correct the Angle (Optional)

If the ChArUco board is not perfectly parallel to the kart, the top-down image will be slightly rotated (or heavily, depending on how good you are at placing the ChArUco board). We can correct this by finding the angle of the center camera and rotating the top-down image. We can simply modify the homography matrix to rotate the image.

#### Step 4.3. Calculate the Region of Interest

The final step is to calculate the region of interest. The top-down image is very large and contains a lot of unnecessary information. We don't need the entire image; instead, we only want to look ahead a certain amount of meters and a certain width.

We know the real-world size of a square on the ChArUco board, which gives us the pixel-to-meter ratio. We can modify the homography matrix to only show a certain region, e.g., 10 meters ahead and 5 meters on each side.

![Region of Interest-min](https://github.com/HeadTriXz/SDC-2024/assets/32986761/1348463b-9ee4-4380-af06-7d2647e7ee59)

## Contributing
Contributions, bug reports, and feature requests are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes and commit them.
4. Push your changes to your fork.
5. Submit a pull request with a description of your changes.

Even if you're not a developer, you can still support the project in other ways. If you find this project useful, consider showing your appreciation by donating to [my Ko-fi page][link-kofi].

[![ko-fi][badge-kofi]][link-kofi]

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [Medium: Multi-View Image Stitching based on the pre-calibrated camera homographies](https://senthillihtnes1994.medium.com/multi-view-image-stitching-based-on-the-pre-calibrated-camera-homographies-991e1fe8a6f4)
- [OpenCV: ArUco Marker Detection](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [OpenCV: Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [OpenCV: Camera Calibration using ChArUco Boards](https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html)
- [OpenCV: Documentation](https://docs.opencv.org/4.x/)
- [StackOverflow: Bird's eye view perspective transformation from camera calibration opencv python](https://stackoverflow.com/questions/48576087/birds-eye-view-perspective-transformation-from-camera-calibration-opencv-python)
- [Wikipedia: Rotation Matrix](https://en.wikipedia.org/wiki/Rotation_matrix)
- [Wikipedia: Vanishing Point](https://en.wikipedia.org/wiki/Vanishing_point)

<!-- Image References -->
[badge-contribute]:https://img.shields.io/badge/contributions-welcome-orange.svg?style=for-the-badge
[badge-kofi]:https://ko-fi.com/img/githubbutton_sm.svg
[badge-license]:https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge
[badge-python]:https://shields.io/badge/python-3.12_|_3.13_|_3.14-blue?style=for-the-badge

<!-- Links -->
[link-kofi]:https://ko-fi.com/headtrixz
[link-sdc]:https://github.com/HeadTriXz/SDC-2024
[link-repo]:https://github.com/HeadTriXz/Multi-Camera-Calibration
