# ml-with-public-knowledge
This code repo belongs to my M.Sc. thesis and is used to back the claimed results.

Two experiments were completed:
1. CoverType dataset experiment. Original Dataset is found here: https://archive-beta.ics.uci.edu/dataset/31/covertype
3. ISOLET dataset experiment. Original Dataset is found here: https://archive-beta.ics.uci.edu/dataset/54/isolet

# Next, detailed the steps needed to run the code for the experiments explained in thesis:

1. For running the experiment on the CoverType dataset:
    * Set "covtype_flag" flag to True
    * Set "isolet_flag" flag to False
    * Set "debug_flag" to False in run_on_dataset_covertype method.
    * Set "debug_flag_entire" to True in run_on_dataset_covertype method to run on the entire dataset size which takes longer to run on locally. Make sure other flags for dataset size are set to False.

2. For running the experiment on the ISOLET dataset:
    * Set "isolet_flag" flag to True
    * Set "covtype_flag" flag to False
    * Set "debug_flag" to False in run_on_dataset_isolet method.
    * The default setting is running on original features columns 37 and 102 separately. Both yielding accuracy improvemnts. Another option which is to conduct a longer experiment by looping through a subset of the 616 existing original features in the dataset and get classification results on each separately. Mind you this is a long wait running on a locally. Just comment/uncomment the required code following the "isolet_flag" if-condition

Arnon.
Department of Computer Science, Ben-Gurion University of the Negev, Beersheva, Israel
