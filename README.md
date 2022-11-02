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
    * Edit "feature_lists" list to contain your "features_list" list of features you want to add to the Elevation-only CoverType dataset. There are 13 new features to select from and for more details check: "for feature_idx in features_list" in the code. The deafult tests are defined in "feature_lists" list.

2. For running the experiment on the ISOLET dataset:
    * Set "isolet_flag" flag to True
    * Set "covtype_flag" flag to False
    * Set "debug_flag" to False in run_on_dataset_isolet method.
    * The default setting is running on original features columns 37 and 102 separately. Both yielding accuracy improvemnts. Another option which is to conduct a longer experiment by looping through a subset of the 616 existing original features in the dataset and get classification results on each separately. Mind you this is a long wait running on a locally. Just comment/uncomment the required code following the "isolet_flag" if-condition
    * The default settings is a test for adding aw_fronting feature which actually consists of 5 columns: aw_fronting_01_column, aw_fronting_02_column, aw_fronting_03_column, aw_fronting_04_column, aw_fronting_05_column. In the case you want to change the added-features you need to comment/uncommet in the code for adding the relevant feature for more details check: "Voicing_Effect_column", "pin_pen_merger_column", "General_American_column" for more options.

Arnon.
Department of Computer Science, Ben-Gurion University of the Negev, Beersheva, Israel
