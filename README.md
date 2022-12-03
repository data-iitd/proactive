# PROACTIVE
Artifacts related to the SIGKDD 2022 paper "Region Invariant Normalizing Flows for Mobility Transfer".

## Requirements
Use a python-3.7 environment and install PyTorch v1.6.0 and TorchVision v0.4.2.

## Execution Instructions
### Dataset Format
We have also provided a dataset used in the paper. For any new dataset, you need to structure it to different files containing the POI details and the mobility details of the users as follows:
```
test_ev.txt test_go.txt test_ti.txt train_ev.txt train_go.txt train_ti.txt
```
### File Details
Here, we provide the description of the files that are given in the dataset:
- train_ev.txt = The types of the actions done by the users.
- train_ti.txt = The times of the actions done by the users.
- train_go.txt = The final goal of the activity.

Similarly, we have the exact three files for the test dataset.

### Running the Code
PROACTIVE requires you to create dumps before running the model. Thus, you must run the develop_dumps.py, for example, run the following command to get pickle dumps for Breakfast dataset:
```
python develop_dumps.py Breakfast
```
The develop_dumps.py is used to combine all the different files, normalize the time of actions, and obtain ".p" dumps for training/testing data.

To train PROACTIVE, use the provided bash script "run.sh". For example, to run the model on Breakfast dataset, use the command:
```
sh run.sh
```
The contents of the script are self-explanatory. Once you run the code, PROACTIVE loads the dataset and predicts the accuracy, MAE, and GPA. The output will of the following format:
![Alt text](bfast.png?raw=true "Output of PROACTIVE")

In detail, the metrics used are as follows
- Acc = The accuracy of predicting the event types.
- MAE = Mean absolute error between the true and predicted action times.
- GPA = The goal prediction accuracy over the test data.
- Itv. GPA = This reflects the ability of PROACTIVE to predict the correct goal at every 'index', i.e, at the incoming of any new action. This is NOT similar to the goal prediction accuracy and is used to keep track of the performance variation due to gamma (RL-trick).

## Citing
If you use this code in your research, please cite:
```
@inproceedings{kdd22,
 author = {Vinayak Gupta and Srikanta Bedathur},
 booktitle = {Proc. of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
 title = {ProActive: Self-Attentive Temporal Point Process Flows for Activity Sequences},
 year = {2022}
}
```

## Contact
In case of any issues, please send a mail to
```guptavinayak51 (at) gmail (dot) com```