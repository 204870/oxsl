# Yet Another Model for Cross Situational Learning
Project for APLN 552: Computational Cognitive Modeling of Language

Cross Situational word learning model, that learns word meaning through vector semantics and object detection

uses the [flicker8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) and [flicker30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) datasets from kaggle

heavily inspired by [this paper](https://aclanthology.org/2014.tal-3.3.pdf) and its [source code](https://github.com/kadarakos/IBMVisual)

tested with the [wordsims3 benchmark](https://github.com/204870/wordsims3)

## How to use

### Creating vectors from images

After downloading your dataset of choice (I reccomend using one of the two linked above), You will need to create a list of images to feed into the preprocessing part of the model. This can easily be done with the unix command `ls | grep jpg >> images.txt`, which I would run once you are in the folder that contains the images in your dataset. Next, make a folder titled `out`, move labeler.py to your current directory and run it after installing the required dependencies (which can be found in requirements.txt).

Once labeler.py has finished running, you'll need to run two more unix commands `ls | grep yolo >> list_yolocats` to combine the filenames of the YOLO model outputs and `ls | grep imagenet >> list_inetcats` for the alexnet outputs. Move vectorize.py to the /out folder created before and run vectorize.py after changing the filename and output variables to whatever you wish, keeping in mind that the output file should have a .mat extension.


footnote: in the future, i want to create a bash script that just does all of this at once.

### Running the model on the new vectors

After creating your vectors in the .mat file, move it to `model/datasets` and then into your dataset of choice (flickr8k or flickr30k). Follow the instructions on the readme and replace the existing `vgg_feats.mat` with the vector file you made. Make sure to rename your file `vgg_feats.mat` in so that the model can recognize it. Once that's all done, head back to the model folder and run `python3 train.py vector online f8k train test`, replacing f8k with your dataset of choice. After running the file, you should see a file `test` which contains the output word vectors.
