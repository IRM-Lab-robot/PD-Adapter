# Inspire Left-Hand Thumb Calibration Card

1. Mount the camera at the fixed calibration viewpoint and do not move it during the three captures.
2. Put the left hand in the calibration fixture or marked table pose with no object.
3. Send three normalized command vectors in this order:
   - open: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
   - half_closed: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
   - thumb_opposition: [1.0, 0.8, 0.2, 0.2, 0.2, 0.2]
4. For each command, wait 1 s after motion stops and take one photo. Save filenames with timestamp and command name.
5. Overlay or compare the observed thumb tip against two FK predictions: mirrored right-hand model and vendor left-hand URDF.
6. Choose the model with smaller thumb-tip image-plane error across the three photos.
7. If the vendor left-hand URDF is closer, estimate a deployment-layer thumb yaw bias by minimizing the three-photo thumb-tip projection error. Record the bias in `execution_manifest.json/global_calibration/thumb_yaw_bias` before running table X trials.
8. If the mirrored model is closer, keep thumb yaw bias at 0 and record `thumb_model_choice=mirrored_model`.
9. Do not run grasp trials until the calibration decision and any bias are written into the manifest copy used by the robot operator.
