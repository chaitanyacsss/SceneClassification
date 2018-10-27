# Scene Classification
1) Classifying images into two labels: indoor and outdoor
2) Used a pretrained resnet as the initial model ('resnet-152') provided by pytorch
3) For training data, some videos were collected from youtube-8M dataset based on the enitity labels provided;
4) The videos are downloaded using pytube and the frames are collected using openCV
5) The labelling is done based on the entity labels of the videos; But some frames manually were removed based on lack of relavence (eg. video title frames etc.) Total dataset has ~ 2010 images.
6) Train/validation/test split is done using a custom dataset class in pytorch, with 200 images in validation set and 200 in the test set.
7) The model is trained for 24 epochs; The training loss, validation loss and the validation set accuracy were plotted real time using tensorboardx.
8) The test accuracy (which can be calculated by running the command "python [run_model.py](run_model.py)" ) is 99.5%
9) Few test cases are added to check the model sanity; check [model_tests.py](model_tests.py)
10) The final model is saved in the checkpoints folder.

To predict the label of a new image using the model, run "python run_model.py -i {path_to_image}"


### NOTE: The saved model is ~200mb and might fail to download when downloading the code. Please check the file size before executing code.
