# ResNet-FPN-Object-Detection
Built a Feature Pyramid Network with a ResNet-50 backbone with a custom Torch loss function for rotated bounding boxes. Used it for object detection and segmentation on adversarial noisy binary images of stars, achieving over 96% IOU accuracy wit under 1.5 Million parameters. Trained on 1 standard Tesla GPU for 5 hours (150 epochs).

Papers cited within code.

The data synthesizer generates images and labels. The model determines whether each image has a star and, if so, finds a rotated bounding box that bounds the star.

More precisely, the labels contain the following five numbers, which the model predicts:
* the x and y coordinates of the center
* yaw
* width and height.

The model is evaluated quantitatively using `compute_score.py`. The metric is the percent of correctly identified stars based on an IOU (Intersection of Union) threshold of 0.7 (for 1024 random samples).
