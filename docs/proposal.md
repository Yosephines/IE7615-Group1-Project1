# Project 1 proposal

Team: Yosephine Tong, Dario Garza and Aditi. IE 7615, Group 1.

## What we are building

We are building a system that identifies which of five celebrities appears in a photo. Later milestones will add the ability to find several faces in one image. We use Python with PyTorch and torchvision to build and train the models.

## Our photos

We use CelebA identity IDs 7007, 2970, 2336, 7 and 4428. Each folder has enough photos to use 23 per person: 17 for learning, 3 for choosing the best model, and 3 for the final test. Using the same number of photos per person keeps the comparison balanced.

## What we found

We compared three image-recognition models. ResNet18 identified the correct person in 10 of 15 test photos (66.7%). The deeper CNN got 6 correct (40.0%), and the small CNN got 5 correct (33.3%).

ResNet18 also performed best on validation. We will use it in the next stages of the project. All trained models and the files needed to check our results are included.

## Team contributions

| Member | Milestone 1 work |
| --- | --- |
| Dario | Prepared the initial data and model-training notebook |
| Aditi | Evaluated models, compared results and shared training logs |
| Yosephine | Organized the repository, repeated training and prepared reports |

## Next steps

Milestone 2 (end of Module 4): combine celebrity photos into new images and label each face with a box. Milestone 3 (mid-Module 5): train YOLOv8 to find those faces. Milestone 4 (end of Module 6): bring the system together and complete the final report.

## Main risks

Our small dataset may not represent the photos the system will see later. Combined images may also look different from real scenes. We will keep training and test images separate and check the face labels carefully.
