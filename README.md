# CameraHardwareImplementation
This is the directory for camera hardware implementation project.

# Circular Marker based Pose estimation.
Compared with original WhyCon system, several changes have been made:
1. support ellipse fitting;
2. implement analytical & geometric position estimation algorithms;
3. add tracking by crop raw image into lots of patches;
4. speedup by parallel computing using OpenMP, real test shows that, for an image of 2000x2000, 50 ms are needed without tracking while only 5ms are needed if tracking is enable.
5. add subpixel refinement;
6. add 3D geometry Levenberg Marquardt Optimization to refine circle's center and major/minor axis's length. Optitrack Test shows improvement.

Things to do:
1. use Decision Tree instead of man-made rules for pattern classification.
2. 6 DOF pose estimation using 2 or more markers.
3. Embedded Implementation on Odroid XU 4.

## Maintainer
1. Xiao Hu
2. Alex G
