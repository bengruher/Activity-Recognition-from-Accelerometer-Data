# Activity-Recognition-from-Accelerometer-Data
Machine learning task to predict participant's activity given data produced by an accelerometer worn on their chest. 

**Background:**

Activity recognition is currently being used by manufacturers of smart phones and smart watches to provide accurate health and exercise information for their users. For example, when I am wearing my Apple Watch and I start to run, my watch generally notifies me that it has detected my activity and asks me if I want to start logging an outdoor run (Apple can distinguish between indoor and outdoor runs). My watch is most likely equipped with various sensors, potentially including an accelerometer, that helps it gather information and make intelligent predictions regarding my activity. 

Another possible use of activity recognition could be detecting whether an elderly person has fallen. Products such as LifeAlert have given elderly people the ability to call for help in these situations. However, these products are not effective if the user is unconscious. The ability to predict what the user is doing, even if that is laying on the floor after a fall, could be useful for detecting such accidents. Recent updates to the Apple Watch have yielded similar functionality. 

The above two use cases for activity recognition represent only the tip of the iceberg when it comes to applications of this technology. I hope to gain experience using time-series by working on a relevant business problem and active area of innovation.

**Dataset:**

The dataset used for this project was taken from the UCI Machine Learning Repository and has been included in this repository. It can also be accessed from https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+from+Single+Chest-Mounted+Accelerometer. 


**Running the project:**

To download the project, simply clone the git repository:
```
git clone https://github.com/bengruher/Activity-Recognition-from-Accelerometer-Data.git
```

To run the project, open the Jupyter notebook (ActivityRecognition.ipynb) and click "Cells" -> "Run All". This will run all of the cells in the notebook and display the output for each cell. You can also run each individual cell by pressing Shift+Enter on that cell. 

You may notice that the cells take a different amount of time to execute. This notebook was originally developed and run on an ml.t2.medium notebook instance on Amazon Sagemaker. 

**Running the project on Amazon Sagemaker:**

Another notebook (SagemakerProject.ipynb) will also be provided with steps to run the most accurate model in an Amazon Sagemaker environment. The notebook guides you through a variety of Sagemaker-specific steps including configuring and launching a training job, evaluating the results of that training job, deploying your model to an inference endpoint, and tearing down that endpoint as part of the cleanup process. This notebook can be run the same way as any Jupyter notebook but requires the AWS SDK to be installed and for your machine to have the necessary AWS permissions to peform the aforementioned Sagemaker actions. 


**Tags:** machine learning, time-series, activity recognition, accelerometer, classification
