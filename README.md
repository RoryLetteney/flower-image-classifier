# flower-image-classifier
My first image classifier. Built as the final project for my AI Programming with Python course.


## To Use:
- Run `python train.py data_dir --arch --save_dir --learning_rate --hidden_units --epochs --gpu`
  - data_dir is the relative path to your training, validation, and testing data (required)
    - myData > train, valid, test
  - --arch is any pretrained pytorch model that accepts 224x224 images (optional, defaults to vgg16)
  - --save_dir is the relative path to a directory in which checkpoint.pth is saved (optional, defaults to root)
  - --learning_rate is the step size at each iteration (optional, defaults to 0.001)
  - --hidden_units is a list of hidden layer input sizes (optional, defaults to [256, 128])
  - --epochs is the number of iterations in which the model trains (optional, defaults to 5)
  - --gpu is a boolean telling the model to utilize the gpu if CUDA is available (optional, defaults to False)
- Run `python predict.py image_path --checkpoint_path --category_names --top_k --gpu
  - image_path is the relative path for the file you wish to have the model predict (required)
  - --checkpoint_path is the relative path for the model state you wish to use for the prediction (optional, defaults to ./checkpoint.pth)
  - --category_names is the relative path to the json file with category to name pairing (optional, defaults to ./cat_to_name.json)
  - --top_k is an integer denoting how many of the top class probabilities should be returned (optional, defaults to 5)
  - --gpu is a boolean telling the model to utilize the gpu if CUDA is available (optional, defaults to False)
