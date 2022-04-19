# SYDE675_Project
## Directories
The repository is composed of several directories. The explanation of each directory is as below:
* Deep Learning‑Based Natural Language Processing in Radiology: The Impact of Report Complexity, Disease Prevalence, Dataset Size, and Algorithm Type on Model Performance
* The original code for implementing the DL codes from the paper as a .pdf file: DLModels-OriginalCodePaper.pdf

* DL Models (Paper code implementation)
  * A python file (.py) file for implementing Neural Network models (Dense, CNN, and LSTM) on the 2 Dataset: Project-deepLearning-NN.py
  
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
       

  
  * BERT Directory including:
    * A python file (.py) file to implement the BERT model on the 2 Dataset: BERT-model.py
    * A dutch pre-trained model downloaded from:
    
    
* ML models
  * A Google Colab file (.ipynb) to implement the Machine Learning models (Naive Bayes, SVM, and Random Forest) on the 2 Dataset
  (The figures and resutls are shown in the Google Colab file)
  
* Paper
 * A pdf file which is the paper that was the motivation of this project: 
 * A pdf file including the original codes to implement the DL models on dataset downloaded from the paper: DLModels-OriginalCodePaper.pdf
  * Dataset
   * Two Dataset as .xlsx files:
     * Data1.xlsx: Chest Radiographs
     * Data2.xlsx: Fracture Radiographs


## Technologies
Project is created with:
* Pycharm version: 3.8
* Google Colab
