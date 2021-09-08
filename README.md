# machine_learning
This directory contains python programs and latex notes on books `Hands-On-Machine Learning with Scikit-Learn, Keras & TensorFlow`, and `Hands on Deep Learning`.

## usage
The main propose of the directory is to make notes of the two referenced books, and support with implementations and experiments on tutorials from the books. 

To run any experiment python code, execute under root directory. Otherwise there might be 'file not found' error

To submit a program to Imperial College London GPU cluster, run `sbatch submit.sh FILE.py` under root directory. The training outputs will be saved in the root directory. 

## file structure
    /machine_learning
    |
    |____/machine-learning_note
    |    | 
    |    |____/note_images (images used in notes)
    |    |____machine-learning_note笔记.pdf (latex note based on `Hands-On-Machine Learning with Scikit-Learn, Keras & TensorFlow`)
    |    |____machine-learning_note笔记.tex (latex source code for the note)
    |
    |____/panda_note (note on pandas python package's usage)
    |
    |____/torch_note (note on torch python framework's usage)
    |
    |____/utils (helper functions used in multiple machine-learning programs)
    |
    |____/logistic_regression 
    |    |
    |    |____basic_logit_regression.py (logit regression based on small, self-defined dataset)
    |    |____gpu_trial.py (trial program submitted to Imperial College London GPU clousters for GPU availability test)
    |
    |____/neural_network (programs about deep learning, based on examples from `Hands on Deep Learning`)
    | 
    |____/SVM (programs about Support Vector Machine,)
    |    | 
    |    |____QP_trial.py (trial on using `quadprog` quadratic problem solver framework)
    |    |____svm_kernel_tutorial.py (SVM classifier using kernel trick)
    |    |____svm_linear_tutorial.py (SVM classifier using linear function as classification boundary)
    | 
    |____/reinforce_learning
    |    | 
    |    |____neural_net_tutorial.py (sudo code on reinforce learning based on network policy)
    |
    |____/parameter_log (directory containing programs' outputs or training data, can be used to transfer parameter between online GPU and local computer)
    |
    |____/training_reports 
    |    | 
    |    |____/raining_oak_style_transfer (summaries of style transfer results and GPU slurm outputs, based on `training_data/images/rainier.jpg` and `training_data/images/autumn_oak.jpg`)
    |
    |____submit.sh (script for submitting a job to IC GPU cluster) 

## Reference
Chinese translation of `Hands-On Machine Learning` https://docsplayer.com/docview/105/168276284/#file=/storage/105/168276284/168276284.pdf 

Only created and used for personal practise and reference, not for any commercial propose. 
