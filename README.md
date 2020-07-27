# P3_revision

Pipeline:
hello

Some cleaned up version of my last paper.

Updates on the code:

In ./Routines:

OriginalDataProcessingFinal.ipynb => AllOriginal.csv Performs some basic cuts to the original asteroseismic data.

makeTraining&TestingSets.ipynb => ../train_data/TrainingOriginal.csv; ../test_data/Test_Original.csv Picks 80% of the data as training; the remainder as testing.

TrainingDataPrepFinal => ../train_data/AllTrainedNorm.csv (will be fed into Model B: BNN_logAge_AllTrainedNorm_modelB.py; same for logDist: BNN_logDist_AllTrainedNorm_modelB.py) Normalizes the training set.

TrainingDataAugmentationFinal => ../HBNN_train_data/AllTrainedNormAugShufffled.csv (will be fed into Model A: BNN_logAge_AllTrainedNormAandS_modelA.py) Performs the distance shuffling and augmentation. The codeTrainingDataAugmentationFinal.py is used to construct five realisations of this dataset to be fed into the final dHBNN Model: dHBNN_logAge_AllTrainingAandS.py

TestDataPrepFinal.ipynb => ../test_data/TestOriginalNorm.csv Normalizes the test data.

OnlyDistanceShufflingTrainingOriginal.ipynb => ../train_data/TrainingOriginalDistanceShuffled.csv; will be fed into Model C: BNN_logDist_TrainedOriginalDistShuffled_modelC.py
