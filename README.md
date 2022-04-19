# SYDE675_Project
## Directories
The repository is composed of several directories. The explanation of each directory is as below:


* DLModels (Paper code implementation):

  * A python file (.py) file for implementing Neural Network models (Dense, CNN, and LSTM) on the 2 Dataset: Project-deepLearning-NN.py
  
  * Results:
    * Results for the Chest Dataset
      * Figures
      * .xlsx files including:
        *  df_neg_TRAIN_THORAX_20201006: training set with negative labels  
        *  df_pos_TRAIN_THORAX_20201006: training set with positive labels
        *  df_TEST_THORAX_20201006: test set
        *  Evaluation_Chest: summary of evaluations for the 3 NN models (Dense, CNN, LSTM)
   
    * Results for the Fracture Dataset
      * Figures
      * .xlsx files including:
        *  df_neg_TRAIN_THORAX_20201006: training set with negative labels  
        *  df_pos_TRAIN_THORAX_20201006: training set with positive labels
        *  df_TEST_THORAX_20201006: test set
        *  Evaluation_Chest: summary of evaluations for the 3 NN models (Dense, CNN, LSTM)
       

  
  
  * A python file (.py) file to implement the BERT model on the 2 Dataset: BERT-model.py
    
    
    
* MLmodels:
  * A Google Colab file (.ipynb) to implement the Machine Learning models (Naive Bayes, SVM, and Random Forest) on the Chest Dataset
  (The figures and resutls are shown in the Google Colab file): MLModels_ChestRadiographs.ipynb
  
  * A Google Colab file (.ipynb) to implement the Machine Learning models (Naive Bayes, SVM, and Random Forest) on the Chest Dataset
  (The figures and resutls are shown in the Google Colab file): MLModels_FractureRadiographs.ipynb
  
* Paper:
   * A pdf file which is the paper that was the motivation of this project: Deep Learning-Based Natural Language Processing in Radiology_ The Impact of   Report Complexity, Disease Prevalence, Dataset Size, and Algorithm Type on Model Performance _ Enhanced Reader.pdf
   * A pdf file including the original codes to implement the DL models on dataset downloaded from the paper: DLModels-OriginalCodePaper.pdf
   * Dataset
     * Two Dataset as .xlsx files:
       * Data1.xlsx: Chest Radiographs
       * Data2.xlsx: Fracture Radiographs


## Technologies
Project is created with:
* Pycharm version: 3.8
* Google Colab
