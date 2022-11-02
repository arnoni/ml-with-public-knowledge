import numpy as np

from Models.decision_tree_model import run_dt
from tabulate import tabulate

import pandas as pd


from datetime import datetime


def option_6_test_on_all_categories(df, debug_flag):  ## Goal_01
    df_copy = df.copy()

    print("------------------------------------------ Option#6 ------------------------------------------------------------------------------")
    print("dataset: train+test")
    print(tabulate(df_copy[0:10], headers='keys', tablefmt='plain'))
    y = df_copy.iloc[:, -1:]
    x = df_copy.iloc[:, :-1]
    res = run_dt(x.values, y.values, debug_flag)
    print("\nReport:")
    print(f"{res}")


def option_test_isolet(df, debug_flag):
    print(
        f"------------------------------------------ option_test_isolet ------------------------------------------------------------------------------")
    df_copy = df.copy()
    print("dataset: train+test")
    print(tabulate(df_copy[0:10], headers='keys', tablefmt='plain'))
    y = df_copy.iloc[:, -1:]
    x = df_copy.iloc[:, :-1]
    res = run_dt(x.values, y.values, debug_flag)
    print("\nReport:")
    print(f"{res}")

def run_on_dataset_isolet(choose_column_indices):


    debug_flag = False

    num_of_datapoints_in_dataset_to_load = 7799

    print("run_on_dataset_isolet - START")




    df_isolet_csv_all5 = pd.read_csv('Datasets/isolet_csv_all5.csv', nrows=num_of_datapoints_in_dataset_to_load, usecols=choose_column_indices,
                                      header=None)


    isolet_01_row_idx_start = 1
    isolet_01_row_idx_end = 1561
    isolet_02_row_idx_start = 1561
    isolet_02_row_idx_end = 3121


    df_isolet_1 = df_isolet_csv_all5.iloc[
        np.r_[isolet_01_row_idx_start:isolet_01_row_idx_end], :]
    df_isolet_2 = df_isolet_csv_all5.iloc[
        np.r_[isolet_02_row_idx_start:isolet_02_row_idx_end], :]




    df_isolet_all_members = pd.read_csv('Datasets/isolet_members_clean_w_state.csv', skip_blank_lines=True, header=None)



    df_isolet_all_members_clean = df_isolet_all_members.dropna(how='all')



    df_isolet1_members = df_isolet_all_members_clean[df_isolet_all_members_clean[0].str.contains('isolet1')]
    print("df_isolet1_members shape: ")
    print(df_isolet1_members.shape)
    df_isolet2_members = df_isolet_all_members_clean[df_isolet_all_members_clean[0].str.contains('isolet2')]
    print("df_isolet2_members shape: ")
    print(df_isolet2_members.shape)

    ## new outside data: dialect_regions_by_us_state
    df_dialect_regions_by_us_state = pd.read_csv('Datasets/dialect_regions_by_us_state.csv', nrows=54)


    df_isolet_1_w_state = add_US_state(1, df_isolet_1, df_isolet1_members, df_dialect_regions_by_us_state,
                                       2)  ## in order to remove USA-tyoe rows from the original dataset
    df_isolet_2_w_state = add_US_state(2, df_isolet_2, df_isolet2_members, df_dialect_regions_by_us_state,
                                       2)  ## in order to remove USA-tyoe rows from the original dataset
    dataset_no_ontologies_data = pd.concat([df_isolet_1_w_state, df_isolet_2_w_state], ignore_index=True)  ## with US state



    dataset_no_ontologies_data = dataset_no_ontologies_data[
    dataset_no_ontologies_data['US_state'] != "None"]


    dataset_no_ontologies_data.drop('US_state', axis=1, inplace=True)


    dataset_no_ontologies_data[617] = dataset_no_ontologies_data[617].astype(str).str.replace("[']", "", regex=True)

    dataset_no_ontologies_data.reset_index()


    df_isolet_1_w_ontologies = add_ontologies(1, df_isolet_1, df_isolet1_members, df_dialect_regions_by_us_state, 2)
    df_isolet_2_w_ontologies = add_ontologies(2, df_isolet_2, df_isolet2_members, df_dialect_regions_by_us_state, 2)


    dataset_with_ontologies_data = pd.concat([df_isolet_1_w_ontologies, df_isolet_2_w_ontologies], ignore_index=True)
    dataset_with_ontologies_data[617] = dataset_with_ontologies_data[617].astype(str).str.replace("[']", "", regex=True)

    dataset_with_ontologies_data.dropna(inplace=True)

    dataset_with_ontologies_data.reset_index()

    ## new feature#01
    # gender_column = df.apply(lambda row: convert_member_id_to_gender(row, df_isolet5_members), axis=1)

    ## new feature#02
    # age_column = df.apply(lambda row: convert_member_id_to_age(row, df_isolet5_members), axis=1)

    ## new feature#03
    # North_Midland_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_01_North_Midland(convert_member_id_to_state(row, df_isolet5_members),
    #                                                              df_dialect_regions_by_us_state), axis=1)

    ## new feature#04
    # South_Midland_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_02_South_Midland(convert_member_id_to_state(row, df_isolet5_members),
    #                                                              df_dialect_regions_by_us_state), axis=1)

    ## new feature#05
    # Inland_North_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_03_Inland_North(convert_member_id_to_state(row, df_isolet5_members),
    #                                                             df_dialect_regions_by_us_state), axis=1)

    ## new feature#06
    # Mid_Atlantic_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_04_Mid_Atlantic(convert_member_id_to_state(row, df_isolet5_members),
    #                                                             df_dialect_regions_by_us_state), axis=1)

    ## new feature#07
    # Eastern_New_England_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_05_Eastern_New_England(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    ## new feature#08
    # Pacific_Northwest_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_06_Pacific_Northwest(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    ## new feature#09
    # Southern_American_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_07_Southern_American(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    ## new feature#10
    # New_York_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_08_New_York(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    ## new feature#11
    # Colorado_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_09_Colorado(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    ## new feature#12
    # California_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_10_California(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    ## new feature#13
    # glide_deletion_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_11_glide_deletion(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    ## new feature#14
    # North_Central_American_English_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_12_North_Central_American_English(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    ## new feature#15
    # cot_caught_merger_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_13_cot_caught_merger(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    ## new feature#16
    ## related to ow_fronting:
    ##majority_ow_fronting_F2_is_above_1400_Hz
    # ow_fronting_01_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_14_ow_fronting_01(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    ##majority_ow_fronting_F2_is_less_than_1400_Hz_above_1300_Hz
    # ow_fronting_02_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_15_ow_fronting_02(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    ##majority_ow_fronting_F2_is_less_than_1300_Hz_above_1200_Hz
    # ow_fronting_03_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_16_ow_fronting_03(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    ##majority_ow_fronting_F2_is_less_than_1200_Hz_above_1100_Hz
    # ow_fronting_04_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_17_ow_fronting_04(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    ##majority_ow_fronting_F2_is_less_than_1100_Hz
    # ow_fronting_05_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_18_ow_fronting_05(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    # pin_pen_merger_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_19_pin_pen_merger(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    # General_American_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_20_General_American(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    # Mid_Southern_2_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_21_Mid_Southern_2(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    # New_English_2_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_22_New_English_2(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    # Voicing_Effect_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_23_Voicing_Effect_column(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    # Oregon_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_24_Oregon_column(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    # Oklahoma_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_25_Oklahoma_column(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    # New_Hampshire_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_26_New_Hampshire_column(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    # Washington_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_27_Washington_column(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    # Illinois_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_28_Illinois_column(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    # Michigan_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_29_Michigan_column(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    # New_Jersey_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_30_New_Jersey_column(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    # Washington_D_C_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_31_Washington_D_C_column(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)


    # df_shape1 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape1[0]
    # df_num_columns1 = df_shape1[1]
    # # print(tabulate(dataset_with_ontologies_data, headers='keys', tablefmt='plain'))
    # dataset_with_ontologies_data.insert(df_num_columns1-1, 'gender', gender_column)

    # df_shape2 = dataset_with_ontologies_data.shape  ## US_region_feature
    # df_num_rows = df_shape2[0]
    # df_num_columns2 = df_shape2[1]
    # dataset_with_ontologies_data.insert(df_num_columns2 - 1, 'is from North Midland US?', North_Midland_column)

    # df_shape3 = dataset_with_ontologies_data.shape  ## US_region_feature
    # df_num_rows = df_shape3[0]
    # df_num_columns3 = df_shape3[1]
    # dataset_with_ontologies_data.insert(df_num_columns3 - 1, 'is from South Midland US?', South_Midland_column)

    # df_shape4 = dataset_with_ontologies_data.shape   ## US_region_feature
    # df_num_rows = df_shape4[0]
    # df_num_columns4 = df_shape4[1]
    # dataset_with_ontologies_data.insert(df_num_columns4 - 1, 'is from Inland North US?', Inland_North_column)

    # df_shape5 = dataset_with_ontologies_data.shape   ## US_region_feature
    # df_num_rows = df_shape5[0]
    # df_num_columns5 = df_shape5[1]
    # dataset_with_ontologies_data.insert(df_num_columns5 - 1, 'is from Mid Atlantic US?', Mid_Atlantic_column)

    # df_shape6 = dataset_with_ontologies_data.shape   ## US_region_feature
    # df_num_rows = df_shape6[0]
    # df_num_columns6 = df_shape6[1]
    # dataset_with_ontologies_data.insert(df_num_columns6 - 1, 'is from Eastern New England US?', Eastern_New_England_column)

    # df_shape7 = dataset_with_ontologies_data.shape  ## US_region_feature
    # df_num_rows = df_shape7[0]
    # df_num_columns7 = df_shape7[1]
    # dataset_with_ontologies_data.insert(df_num_columns7 - 1, 'is from Pacific Northwest US?', Pacific_Northwest_column)

    # df_shape8 = dataset_with_ontologies_data.shape ## US_region_feature
    # df_num_rows = df_shape8[0]
    # df_num_columns8 = df_shape8[1]
    # dataset_with_ontologies_data.insert(df_num_columns8 - 1, 'is from Southern American US?', Southern_American_column)

    # df_shape9 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape9[0]
    # df_num_columns9 = df_shape9[1]
    # dataset_with_ontologies_data.insert(df_num_columns9 - 1, 'is from New York US?', New_York_column)

    # df_shape10 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape10[0]
    # df_num_columns10 = df_shape10[1]
    # dataset_with_ontologies_data.insert(df_num_columns10 - 1, 'is from Colorado US?', Colorado_column)

    # df_shape11 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape11[0]
    # df_num_columns11 = df_shape11[1]
    # dataset_with_ontologies_data.insert(df_num_columns11 - 1, 'is from California US?', California_column)

    # df_shape12 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape12[0]
    # df_num_columns12 = df_shape12[1]
    # dataset_with_ontologies_data.insert(df_num_columns12 - 1, 'glide deletion level', glide_deletion_column)

    # df_shape13 = dataset_with_ontologies_data.shape  ## US_region_feature
    # df_num_rows = df_shape13[0]
    # df_num_columns13 = df_shape13[1]
    # dataset_with_ontologies_data.insert(df_num_columns13 - 1, 'is from North-Central American English US?', North_Central_American_English_column)

    # df_shape14 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape14[0]
    # df_num_columns14 = df_shape14[1]
    # dataset_with_ontologies_data.insert(df_num_columns14 - 1, 'cot_caught_merger',
    #                                     cot_caught_merger_column)

    ## ow_fronting
    ##  related to ow_fronting:
    # df_shape14 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape14[0]
    # df_num_columns14 = df_shape14[1]
    # dataset_with_ontologies_data.insert(df_num_columns14 - 1, 'majority_ow_fronting_F2_is_above_1400_Hz',
    #                                     ow_fronting_01_column)

    # df_shape15 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape15[0]
    # df_num_columns15 = df_shape15[1]
    # dataset_with_ontologies_data.insert(df_num_columns15 - 1, 'majority_ow_fronting_F2_is_less_than_1400_Hz_above_1300_Hz',
    #                                     ow_fronting_02_column)

    # df_shape16 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape16[0]
    # df_num_columns16 = df_shape16[1]
    # dataset_with_ontologies_data.insert(df_num_columns16 - 1,
    #                                     'majority_ow_fronting_F2_is_less_than_1300_Hz_above_1200_Hz',
    #                                     ow_fronting_03_column)

    # df_shape17 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape17[0]
    # df_num_columns17 = df_shape17[1]
    # dataset_with_ontologies_data.insert(df_num_columns17 - 1,
    #                                     'majority_ow_fronting_F2_is_less_than_1200_Hz_above_1100_Hz',
    #                                     ow_fronting_04_column)

    # df_shape18 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape18[0]
    # df_num_columns18 = df_shape18[1]
    # dataset_with_ontologies_data.insert(df_num_columns18 - 1,
    #                                     'majority_ow_fronting_F2_is_less_than_1100_Hz',
    #                                     ow_fronting_05_column)

    # df_shape19 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape19[0]
    # df_num_columns19 = df_shape19[1]
    # dataset_with_ontologies_data.insert(df_num_columns19 - 1,
    #                                     'pin_pen_merger',
    #                                     pin_pen_merger_column)

    # df_shape20 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape20[0]
    # df_num_columns20 = df_shape20[1]
    # dataset_with_ontologies_data.insert(df_num_columns20 - 1,
    #                                     'General_American',
    #                                     General_American_column)

    # df_shape21 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape21[0]
    # df_num_columns21 = df_shape21[1]
    # dataset_with_ontologies_data.insert(df_num_columns21 - 1,
    #                                     'Mid_Southern_2',
    #                                     Mid_Southern_2_column)

    # df_shape22 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape22[0]
    # df_num_columns22 = df_shape22[1]
    # dataset_with_ontologies_data.insert(df_num_columns22 - 1,
    #                                     'New_English_2',
    #                                     New_English_2_column)

    # df_shape23 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape23[0]
    # df_num_columns23 = df_shape23[1]
    # dataset_with_ontologies_data.insert(df_num_columns23 - 1,
    #                                     'Voicing_Effect',
    #                                     Voicing_Effect_column)

    # df_shape24 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape24[0]
    # df_num_columns24 = df_shape24[1]
    # dataset_with_ontologies_data.insert(df_num_columns24 - 1,
    #                                     'Oregon',
    #                                     Oregon_column)

    # df_shape25 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape25[0]
    # df_num_columns25 = df_shape25[1]
    # dataset_with_ontologies_data.insert(df_num_columns25 - 1,
    #                                     'Oklahoma',
    #                                     Oklahoma_column)

    # df_shape25 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape25[0]
    # df_num_columns25 = df_shape25[1]
    # dataset_with_ontologies_data.insert(df_num_columns25 - 1,
    #                                     'New_Hampshire',
    #                                     New_Hampshire_column)

    # df_shape26 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape26[0]
    # df_num_columns26 = df_shape26[1]
    # dataset_with_ontologies_data.insert(df_num_columns26 - 1,
    #                                     'Washington',
    #                                     Washington_column)

    # df_shape27 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape27[0]
    # df_num_columns27 = df_shape27[1]
    # dataset_with_ontologies_data.insert(df_num_columns27 - 1,
    #                                     'Illinois',
    #                                     Illinois_column)

    # df_shape28 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape28[0]
    # df_num_columns28 = df_shape28[1]
    # dataset_with_ontologies_data.insert(df_num_columns28 - 1,
    #                                     'Michigan',
    #                                     Michigan_column)

    # df_shape29 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape29[0]
    # df_num_columns29 = df_shape29[1]
    # dataset_with_ontologies_data.insert(df_num_columns29 - 1,
    #                                     'New_Jersey',
    #                                     New_Jersey_column)

    # df_shape30 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape30[0]
    # df_num_columns30 = df_shape30[1]
    # dataset_with_ontologies_data.insert(df_num_columns30 - 1,
    #                                     'Washington_D_C',
    #                                     Washington_D_C_column)

    # move label back to be in the last column:
    # column_to_reorder = dataset_with_ontologies_data.pop('target_class')
    dataset_with_ontologies_data.head()
    print(dataset_with_ontologies_data.columns.values.tolist())
    column_to_reorder = dataset_with_ontologies_data.pop(617)
    dataset_with_ontologies_data.insert(len(dataset_with_ontologies_data.columns), 'Alphabet letters',
                                        column_to_reorder)


    column_to_reorder = dataset_no_ontologies_data.pop(617)
    dataset_no_ontologies_data.insert(len(dataset_no_ontologies_data.columns), 'Alphabet letters',
                                        column_to_reorder)

    print("option_test_isolet : no ontologies! - START")
    option_test_isolet(dataset_no_ontologies_data, debug_flag)  ## Goal_01
    print("option_test_isolet : no ontologies! - FINISH")

    print("\n\noption_test_isolet : with ontologies! - START")
    option_test_isolet(dataset_with_ontologies_data, debug_flag)  ## Goal_01
    print("option_test_isolet : with ontologies! - FINISH")

def add_US_state(isolet_idx, df_isolet_x, df_isolet_members, df_dialect_regions_by_us_state, option_isolet5):
    print('add_US_state - START - only to original (no ontology) dataset')
    df = df_isolet_x.copy()

    us_state_column = df.apply(
        lambda row: convet_member_id_to_us_state(
            convert_member_id_to_state(row, df_isolet_members, isolet_idx, option_isolet5),
            df_dialect_regions_by_us_state), axis=1)
    df_num_columns = df.shape[1]
    df.insert(df_num_columns - 1, 'US_state', us_state_column)
    return df

def add_ontologies(isolet_idx, df_isolet_x, df_isolet_members, df_dialect_regions_by_us_state, option_isolet5):
    print('add_ontologies - START')
    df = df_isolet_x.copy()

##15 in my ISOLET dev notes:
#     General_American_column = df.apply(
#         lambda row: convet_member_id_to_dialect_20_General_American(
#             convert_member_id_to_state(row, df_isolet_members, isolet_idx,option_isolet5),
#             df_dialect_regions_by_us_state), axis=1)
# ##16 in my ISOLET dev notes: Lowland_Southern
#     Lowland_Southern_column = df.apply(
#         lambda row: convet_member_id_to_dialect_Lowland_Southern(
#             convert_member_id_to_state(row, df_isolet_members, isolet_idx,option_isolet5),
#             df_dialect_regions_by_us_state), axis=1)
##17 in my ISOLET dev notes: Mid_Southern_1


#     Mid_Southern_1_column = df.apply(
#         lambda row: convet_member_id_to_dialect_Mid_Southern_1(
#             convert_member_id_to_state(row, df_isolet_members, isolet_idx,option_isolet5),
#             df_dialect_regions_by_us_state), axis=1)
# ##18 in my ISOLET dev notes: Mid_Southern_2
#     Mid_Southern_2_column = df.apply(
#         lambda row: convet_member_id_to_dialect_21_Mid_Southern_2(
#             convert_member_id_to_state(row, df_isolet_members, isolet_idx,option_isolet5),
#             df_dialect_regions_by_us_state), axis=1)
# #19 New_English_1
#     New_English_1_column = df.apply(
#         lambda row: convet_member_id_to_dialect_22_New_English_2(
#             convert_member_id_to_state(row, df_isolet_members, isolet_idx,option_isolet5),
#             df_dialect_regions_by_us_state), axis=1)

    # 20 New_English_2 - no need for ISOLET_1 and ISOLET_2
    # New_English_2_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_22_New_English_2(
    #         convert_member_id_to_state(row, df_isolet5_members),
    #         df_dialect_regions_by_us_state), axis=1)

    # glide_deletion_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_11_glide_deletion(
    #         convert_member_id_to_state(row, df_isolet_members, isolet_idx,option_isolet5),
    #         df_dialect_regions_by_us_state), axis=1)

    # us_state_column = df.apply(
    #     lambda row: convet_member_id_to_us_state(
    #         convert_member_id_to_state(row, df_isolet_members, isolet_idx, option_isolet5),
    #         df_dialect_regions_by_us_state), axis=1)

    # pin_pen_merger_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_19_pin_pen_merger(
    #         convert_member_id_to_state(row, df_isolet_members, isolet_idx, option_isolet5),
    #         df_dialect_regions_by_us_state), axis=1)

    # Voicing_Effect_column = df.apply(
    #     lambda row: convet_member_id_to_dialect_23_Voicing_Effect_column(
    #         convert_member_id_to_state(row, df_isolet_members, isolet_idx, option_isolet5),
    #         df_dialect_regions_by_us_state), axis=1)


    aw_fronting_01_column = df.apply(
        lambda row: convet_member_id_to_dialect_majority_aw_fronting_01_column(
            convert_member_id_to_state(row, df_isolet_members, isolet_idx, option_isolet5),
            df_dialect_regions_by_us_state), axis=1)

    aw_fronting_02_column = df.apply(
        lambda row: convet_member_id_to_dialect_majority_aw_fronting_02_column(
            convert_member_id_to_state(row, df_isolet_members, isolet_idx, option_isolet5),
            df_dialect_regions_by_us_state), axis=1)
    aw_fronting_03_column = df.apply(
        lambda row: convet_member_id_to_dialect_majority_aw_fronting_03_column(
            convert_member_id_to_state(row, df_isolet_members, isolet_idx, option_isolet5),
            df_dialect_regions_by_us_state), axis=1)
    aw_fronting_04_column = df.apply(
        lambda row: convet_member_id_to_dialect_majority_aw_fronting_04_column(
            convert_member_id_to_state(row, df_isolet_members, isolet_idx, option_isolet5),
            df_dialect_regions_by_us_state), axis=1)
    aw_fronting_05_column = df.apply(
        lambda row: convet_member_id_to_dialect_majority_aw_fronting_05_column(
            convert_member_id_to_state(row, df_isolet_members, isolet_idx, option_isolet5),
            df_dialect_regions_by_us_state), axis=1)

    df_num_columns = df.shape[1]
    df.insert(df_num_columns - 1, 'aw_fronting_01', aw_fronting_01_column)
    df_num_columns = df.shape[1]
    df.insert(df_num_columns - 1, 'aw_fronting_02', aw_fronting_02_column)
    df_num_columns = df.shape[1]
    df.insert(df_num_columns - 1, 'aw_fronting_03', aw_fronting_03_column)
    df_num_columns = df.shape[1]
    df.insert(df_num_columns - 1, 'aw_fronting_04', aw_fronting_04_column)
    df_num_columns = df.shape[1]
    df.insert(df_num_columns - 1, 'aw_fronting_05', aw_fronting_05_column)

    # df.insert(df_num_columns - 1, 'Voicing_Effect',     Voicing_Effect_column)
    # df_num_columns = df.shape[1]
    # df.insert(df_num_columns - 1, 'glide deletion level', glide_deletion_column)
    # df_num_columns = df.shape[1]
    # df.insert(df_num_columns - 1, 'US_state', us_state_column)
    # df.insert(df_num_columns - 1, 'New_English_1', New_English_1_column)
    # df_num_columns = df.shape[1]
    # df.insert(df_num_columns - 1, 'Mid_Southern_2', Mid_Southern_2_column)
    # df_num_columns = df.shape[1]
    # df.insert(df_num_columns - 1, 'Mid_Southern_1', Mid_Southern_1_column)
    # df_num_columns = df.shape[1]
    # df.insert(df_num_columns - 1, 'Lowland_Southern', Lowland_Southern_column)
    # df_num_columns = df.shape[1]
    # df.insert(df_num_columns - 1, 'General_American', General_American_column)

    return df


def run_on_dataset_covertype(choose_column_indices):

    debug_flag = True

    debug_flag_very_small = False
    debug_flag_small = False
    debug_flag_small2 = False
    debug_flag_small3 = True
    debug_flag_medium = False
    debug_flag_half = False
    debug_flag_entire = False


    if debug_flag_very_small:
        num_of_rows_in_dataset_in_train_test = 100
    elif debug_flag_small:
        num_of_rows_in_dataset_in_train_test = 2000
    elif debug_flag_small2:
        num_of_rows_in_dataset_in_train_test = 10000
    elif debug_flag_small3:
        num_of_rows_in_dataset_in_train_test = 50000
    elif debug_flag_medium:
        num_of_rows_in_dataset_in_train_test = 100000
    elif debug_flag_half:
        num_of_rows_in_dataset_in_train_test = 290506
    elif debug_flag_entire:
        num_of_rows_in_dataset_in_train_test = 581012


    print("run_on_dataset_covertype - START")
    print("run_on_dataset_covertype - START")

    #                  header=None)
    df_all = pd.read_csv('Datasets/covtype.csv', nrows=581012, usecols=choose_column_indices,
                     header=None)
    df = df_all.sample(num_of_rows_in_dataset_in_train_test)
    df_shape = df.shape
    df_num_rows = df_shape[0]
    df_num_columns = df_shape[1]
    print(f"df_num_columns = {df_num_columns}")

    df_soil_features_2022_08_11 = pd.read_csv(
        'Datasets/new_soil_type_features_2022_08_11.csv')

    df.rename(columns={0: "Elevation"}, inplace=True)
    df.rename(columns={54: "Cover_type"}, inplace=True)
    df.describe()

    print(tabulate(df[0:10], headers='keys', tablefmt='plain'))
    dataset_no_ontologies_data = df

    ## new feature#01
    water_storage_column = df.apply(
        lambda row: convet_soil_type_to_water_storage(get_soil_type_row_idx(row), df_soil_features_2022_08_11), axis=1)
    ## new feature#02
    water_percentage_column = df.apply(
        lambda row: convet_soil_type_to_water_percentage(get_soil_type_row_idx(row), df_soil_features_2022_08_11),
        axis=1)
    ## new feature#03
    dominant_geomorphic_position_column = df.apply(
        lambda row: convet_soil_type_to_dominant_geomorphic_position(get_soil_type_row_idx(row),
                                                                     df_soil_features_2022_08_11), axis=1)
    ## new feature#04
    rubble_land_percentage_column = df.apply(
        lambda row: convet_soil_type_to_rubble_land_percentage(get_soil_type_row_idx(row), df_soil_features_2022_08_11),
        axis=1)

    ## new feature#05
    rock_outcrop_percentage_column = df.apply(
        lambda row: convet_soil_type_to_rock_outcrop_percentage(get_soil_type_row_idx(row),
                                                                df_soil_features_2022_08_11), axis=1)

    ## new feature#06
    total_plant_available_water_1st_column = df.apply(
        lambda row: convet_soil_type_to_total_plant_available_water_1st(get_soil_type_row_idx(row),
                                                                        df_soil_features_2022_08_11), axis=1)

    ## new feature#07
    organic_carbon_stock_0_100_cm_low_1st_column = df.apply(
        lambda row: convet_soil_type_to_organic_carbon_stock_0_100_cm_low_1st(get_soil_type_row_idx(row),
                                                                              df_soil_features_2022_08_11),
        axis=1)

    ## new feature#08
    organic_carbon_stock_0_100_cm_high_1st_column = df.apply(
        lambda row: convet_soil_type_to_organic_carbon_stock_0_100_cm_high_1st(get_soil_type_row_idx(row),
                                                                               df_soil_features_2022_08_11),
        axis=1)

    ## new feature#09
    T_erosion_factor_1st_column = df.apply(
        lambda row: convet_soil_type_to_T_erosion_factor_1st(get_soil_type_row_idx(row),
                                                             df_soil_features_2022_08_11),
        axis=1)

    ## new feature#10
    total_plant_available_water_2nd_column = df.apply(
        lambda row: convet_soil_type_to_total_plant_available_water_2nd(get_soil_type_row_idx(row),
                                                                        df_soil_features_2022_08_11),
        axis=1)

    ## new feature#11
    organic_carbon_stock_0_100_cm_low_2nd_column = df.apply(
        lambda row: convet_soil_type_to_organic_carbon_stock_0_100_cm_low_2nd(get_soil_type_row_idx(row),
                                                                              df_soil_features_2022_08_11),
        axis=1)

    ## new feature#12
    organic_carbon_stock_0_100_cm_high_2nd_column = df.apply(
        lambda row: convet_soil_type_to_organic_carbon_stock_0_100_cm_high_2nd(get_soil_type_row_idx(row),
                                                                               df_soil_features_2022_08_11),
        axis=1)

    ## new feature#13
    T_erosion_factor_2nd_column = df.apply(
        lambda row: convet_soil_type_to_T_erosion_factor_2nd(get_soil_type_row_idx(row),
                                                             df_soil_features_2022_08_11),
        axis=1)

    drop_list_int = list(range(10, 54, 1))
    drop_columns_list_string = map(str, drop_list_int)
    df.drop(drop_list_int, inplace=True, axis=1)
    print(tabulate(df[0:10], headers='keys', tablefmt='plain'))


    dataset_with_ontologies_data = df.copy()
    df_shape1 = dataset_with_ontologies_data.shape
    df_num_rows = df_shape1[0]
    df_num_columns1 = df_shape1[1]
    # print(tabulate(dataset_with_ontologies_data, headers='keys', tablefmt='plain'))
    # dataset_with_ontologies_data.insert(df_num_columns1-1, 'water_storage', water_storage_column)
    # outside_feature_added_name = "water_storage"

    df_shape2 = dataset_with_ontologies_data.shape
    df_num_rows = df_shape2[0]
    df_num_columns2 = df_shape2[1]
    print(f"df_num_columns2 = {df_num_columns}")

    # dataset_with_ontologies_data.insert(df_num_columns2 - 1, 'water_percentage', water_percentage_column)
    # outside_feature_added_name = "water_percentage"

    # df_shape3 = dataset_with_ontologies_data.shape
    # df_num_rows = df_shape3[0]
    # df_num_columns3 = df_shape3[1]
    # dataset_with_ontologies_data.insert(df_num_columns3 - 1, 'dominant_geomorphic_position',
    #                                     dominant_geomorphic_position_column)
    #
    # dataset_with_ontologies_data = pd.get_dummies(dataset_with_ontologies_data,
    #                                               columns=['dominant_geomorphic_position'])
    # outside_feature_added_name = "dominant_geomorphic_position"

    ## adding new 10 features: START
    df_shape4 = dataset_with_ontologies_data.shape
    df_num_rows = df_shape4[0]
    df_num_columns4 = df_shape4[1]
    # dataset_with_ontologies_data.insert(df_num_columns4 - 1, 'rubble land percentage',
    #                                     rubble_land_percentage_column)
    # outside_feature_added_name = "rubble land percentage"
    ##---------------
    df_shape5 = dataset_with_ontologies_data.shape
    df_num_rows = df_shape5[0]
    df_num_columns5 = df_shape5[1]
    # dataset_with_ontologies_data.insert(df_num_columns5 - 1, 'rock outcrop percentage',
    #                                     rock_outcrop_percentage_column)
    # outside_feature_added_name = "rock outcrop percentage"

    ##---------------
    df_shape6 = dataset_with_ontologies_data.shape
    df_num_rows = df_shape6[0]
    df_num_columns6 = df_shape6[1]
    # dataset_with_ontologies_data.insert(df_num_columns6 - 1, 'total plant available water 1st',
    #                                     total_plant_available_water_1st_column)
    # outside_feature_added_name = "total plant available water 1st"

    ##---------------
    df_shape7 = dataset_with_ontologies_data.shape
    df_num_rows = df_shape7[0]
    df_num_columns7 = df_shape7[1]
    # dataset_with_ontologies_data.insert(df_num_columns7 - 1, 'organic carbon stock 0-100 cm low 1st',
    #                                     organic_carbon_stock_0_100_cm_low_1st_column)
    # outside_feature_added_name = "organic carbon stock 0-100 cm low 1st"

    ##---------------
    df_shape8 = dataset_with_ontologies_data.shape
    df_num_rows = df_shape8[0]
    df_num_columns8 = df_shape8[1]
    # dataset_with_ontologies_data.insert(df_num_columns8 - 1, 'organic carbon stock 0-100 cm high 1st',
    #                                     organic_carbon_stock_0_100_cm_high_1st_column)
    # outside_feature_added_name = "organic carbon stock 0-100 cm high 1st"

    ##---------------
    df_shape9 = dataset_with_ontologies_data.shape
    df_num_rows = df_shape9[0]
    df_num_columns9 = df_shape9[1]
    # dataset_with_ontologies_data.insert(df_num_columns9 - 1, 'T erosion factor 1st',
    #                                     T_erosion_factor_1st_column)
    # outside_feature_added_name = "T erosion factor 1st"

    ##---------------
    df_shape10 = dataset_with_ontologies_data.shape
    df_num_rows = df_shape10[0]
    df_num_columns10 = df_shape10[1]
    # dataset_with_ontologies_data.insert(df_num_columns10 - 1, 'total plant available water 2nd',
    #                                     total_plant_available_water_2nd_column)
    # outside_feature_added_name = "total plant available water 2nd"

    ##---------------
    df_shape11 = dataset_with_ontologies_data.shape
    df_num_rows = df_shape11[0]
    df_num_columns11 = df_shape11[1]
    # dataset_with_ontologies_data.insert(df_num_columns11 - 1, 'organic carbon stock 0-100 cm low 2nd',
    #                                     organic_carbon_stock_0_100_cm_low_2nd_column)
    # outside_feature_added_name = "organic carbon stock 0-100 cm low 2nd"

    ##---------------
    df_shape12 = dataset_with_ontologies_data.shape
    df_num_rows = df_shape12[0]
    # df_num_columns12 = df_shape12[1]
    # dataset_with_ontologies_data.insert(df_num_columns12 - 1, 'organic carbon stock 0-100 cm high 2nd',
    #                                     organic_carbon_stock_0_100_cm_high_2nd_column)
    outside_feature_added_name = "organic carbon stock 0-100 cm high 2nd"

    ##---------------
    df_shape13 = dataset_with_ontologies_data.shape
    df_num_rows = df_shape13[0]
    df_num_columns13 = df_shape13[1]
    # dataset_with_ontologies_data.insert(df_num_columns13 - 1, 'T erosion factor 2nd',
    #                                     T_erosion_factor_2nd_column)
    # outside_feature_added_name = "T erosion factor 2nd"

    ##---------------
    ## adding new 10 features: END

    #ENTER HERE: test features:
    features_list_01 = [12]
    features_list_02 = [11]
    features_list_03 = [10]
    features_list_04 = [9]
    features_list_05 = [8]
    features_list_06 = [7]
    features_list_07 = [6]
    features_list_08 = [5]
    features_list_09 = [4]
    features_list_10 = [3]
    features_list_11 = [2]
    features_list_12 = [1]
    features_list_13 = [13]

    features_list_14 = [12, 11]
    features_list_15 = [11, 10]
    features_list_16 = [9, 5]
    features_list_17 = [12,4]
    features_list_18 = [11,2]
    features_list_19 = [3,7]
    features_list_20 = [7,1]
    features_list_21 = [10,5]
    features_list_22 = [2,13]
    features_list_23 = [13,11]
    features_list_24 = [5,8]
    features_list_25 = [8,3]
    features_list_26 = [6,10]
    features_list_27 = [7,6]
    features_list_28 = [13,6]
    features_list_29 = [12,4]
    features_list_30 = [3,9]


    feature_lists = [features_list_01, features_list_02, features_list_03, features_list_04, features_list_05, features_list_06,
                     features_list_07, features_list_08, features_list_09, features_list_10, features_list_11, features_list_12,
                     features_list_13, features_list_14, features_list_15, features_list_16, features_list_17, features_list_18,
                     features_list_19, features_list_20, features_list_21, features_list_22, features_list_23, features_list_24,
                     features_list_25, features_list_26, features_list_27, features_list_28, features_list_29, features_list_30]


    print("option_6_test_on_all_categories with dataset_no_ontologies_data:")
    option_6_test_on_all_categories(dataset_no_ontologies_data, debug_flag)  ## Goal_01

    for features_list in feature_lists:
        del dataset_with_ontologies_data
        dataset_with_ontologies_data = df.copy()
        outside_feature_added_name = ""
        for feature_idx in features_list:
            print(f"feature_idx = {feature_idx}:")
            df_shape = dataset_with_ontologies_data.shape
            df_num_columns = df_shape[1]


            if feature_idx == 1:
                dataset_with_ontologies_data.insert(df_num_columns-1, 'water_storage', water_storage_column)
                outside_feature_added_name = outside_feature_added_name + ", " + "water_storage"
            if feature_idx == 2:
                dataset_with_ontologies_data.insert(df_num_columns - 1, 'water_percentage', water_percentage_column)
                outside_feature_added_name = outside_feature_added_name + ", " + "water_percentage"
            if feature_idx == 3:
                dataset_with_ontologies_data.insert(df_num_columns - 1, 'dominant_geomorphic_position',
                                                    dominant_geomorphic_position_column)
                dataset_with_ontologies_data = pd.get_dummies(dataset_with_ontologies_data,
                                                              columns=['dominant_geomorphic_position'])

                column_to_reorder = dataset_with_ontologies_data.pop('Cover_type')
                dataset_with_ontologies_data.insert(len(dataset_with_ontologies_data.columns), 'Cover_type',
                                                    column_to_reorder)
                outside_feature_added_name = outside_feature_added_name + ", " + "dominant_geomorphic_position"
            # *************
            if feature_idx == 4:
                dataset_with_ontologies_data.insert(df_num_columns - 1, 'rubble land percentage',
                                                    rubble_land_percentage_column)
                outside_feature_added_name = outside_feature_added_name + ", " + "rubble land percentage"
            if feature_idx == 5:
                dataset_with_ontologies_data.insert(df_num_columns - 1, 'rock outcrop percentage',
                                                    rock_outcrop_percentage_column)
                outside_feature_added_name = outside_feature_added_name + ", " + "rock outcrop percentage"
            if feature_idx == 6:
                dataset_with_ontologies_data.insert(df_num_columns - 1, 'total plant available water 1st',
                                                    total_plant_available_water_1st_column)
                outside_feature_added_name = outside_feature_added_name + ", " + "total plant available water 1st"
            # *************
            if feature_idx == 7:
                dataset_with_ontologies_data.insert(df_num_columns - 1, 'organic carbon stock 0-100 cm low 1st',
                                                    organic_carbon_stock_0_100_cm_low_1st_column)
                outside_feature_added_name = outside_feature_added_name + ", " + "organic carbon stock 0-100 cm low 1st"
            if feature_idx == 8:
                dataset_with_ontologies_data.insert(df_num_columns - 1, 'organic carbon stock 0-100 cm high 1st',
                                                    organic_carbon_stock_0_100_cm_high_1st_column)
                outside_feature_added_name = outside_feature_added_name + ", " + "organic carbon stock 0-100 cm high 1st"
            if feature_idx == 9:
                dataset_with_ontologies_data.insert(df_num_columns - 1, 'T erosion factor 1st',
                                                    T_erosion_factor_1st_column)
                outside_feature_added_name = outside_feature_added_name + ", " + "T erosion factor 1st"
            # *************
            if feature_idx == 10:
                dataset_with_ontologies_data.insert(df_num_columns - 1, 'total plant available water 2nd',
                                                    total_plant_available_water_2nd_column)
                outside_feature_added_name = outside_feature_added_name + ", " + "total plant available water 2nd"
            if feature_idx == 11:
                dataset_with_ontologies_data.insert(df_num_columns - 1, 'organic carbon stock 0-100 cm low 2nd',
                                                    organic_carbon_stock_0_100_cm_low_2nd_column)
                outside_feature_added_name = outside_feature_added_name + ", " + "organic carbon stock 0-100 cm low 2nd"
            if feature_idx == 12:
                dataset_with_ontologies_data.insert(df_num_columns - 1, 'organic carbon stock 0-100 cm high 2nd',
                                                    organic_carbon_stock_0_100_cm_high_2nd_column)
                outside_feature_added_name = outside_feature_added_name + ", " + "organic carbon stock 0-100 cm high 2nd"
            if feature_idx == 13:
                dataset_with_ontologies_data.insert(df_num_columns - 1, 'T erosion factor 2nd',
                                                    T_erosion_factor_2nd_column)
                outside_feature_added_name = outside_feature_added_name + ", " + "T erosion factor 2nd"



        print("------------------------------------------------------------------------------------")
        print(f"option_6_test_on_all_categories with dataset_with_ontologies_data with added features {outside_feature_added_name}:")
        option_6_test_on_all_categories(dataset_with_ontologies_data, debug_flag)  ## Goal_01



def convert_member_id_to_gender(row, df_members):

    row_idx = row.name
    isolet_inside = row_idx % 1560
    isolet_member_idx = isolet_inside // 52
    isolet_member_gender = df_members.iloc[isolet_member_idx, 3]
    isolet_member_id = df_members.iloc[isolet_member_idx, 0]

    return isolet_member_gender


def convert_member_id_to_age(row, df_members):
    row_idx = row.name

    isolet_inside = row_idx % 1560
    isolet_member_idx = isolet_inside // 52

    isolet_member_age = df_members.iloc[isolet_member_idx, 1]

    return isolet_member_age


def convert_member_id_to_state(row, df_members, isolet_idx, option):

    row_idx = row.name

    if (isolet_idx == 5):
        row_idx = row_idx+2

    row_idx = row_idx -1
    isolet_inside = row_idx % 1560
    isolet_member_idx = isolet_inside // 52
    isolet_member_state = df_members.iloc[isolet_member_idx, 2]

    return isolet_member_state


def convet_member_id_to_dialect_01_North_Midland(state, df_dialects):

    state_row = df_dialects.loc[df_dialects['State name'] == state]
    is_North_Midland = state_row.iloc[0]['North Midland']
    return is_North_Midland


def convet_member_id_to_dialect_02_South_Midland(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['South Midland']


def convet_member_id_to_dialect_03_Inland_North(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['Inland North']


def convet_member_id_to_dialect_04_Mid_Atlantic(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['Mid-Atlantic']


def convet_member_id_to_dialect_05_Eastern_New_England(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['Eastern New England']


def convet_member_id_to_dialect_06_Pacific_Northwest(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['Pacific Northwest']


def convet_member_id_to_dialect_07_Southern_American(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['Southern American']


def convet_member_id_to_dialect_08_New_York(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['New York']


def convet_member_id_to_dialect_09_Colorado(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['Colorado']


def convet_member_id_to_dialect_10_California(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['California']


def convet_member_id_to_dialect_11_glide_deletion(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['glide_deletion']


def convet_member_id_to_dialect_12_North_Central_American_English(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['North-Central American English']


def convet_member_id_to_dialect_13_cot_caught_merger(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['cot_caught_merger']


def convet_member_id_to_dialect_14_ow_fronting_01(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['majority_ow_fronting_F2_is_above_1400_Hz']


def convet_member_id_to_dialect_15_ow_fronting_02(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['majority_ow_fronting_F2_is_less_than_1400_Hz_above_1300_Hz']


def convet_member_id_to_dialect_16_ow_fronting_03(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['majority_ow_fronting_F2_is_less_than_1300_Hz_above_1200_Hz']


def convet_member_id_to_dialect_17_ow_fronting_04(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['majority_ow_fronting_F2_is_less_than_1200_Hz_above_1100_Hz']


def convet_member_id_to_dialect_18_ow_fronting_05(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['majority_ow_fronting_F2_is_less_than_1100_Hz']


def convet_member_id_to_dialect_19_pin_pen_merger(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['pin_pen_merger']


def convet_member_id_to_dialect_20_General_American(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['General_American']


def convet_member_id_to_dialect_21_Mid_Southern_2(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['Mid_Southern_2']

def convet_member_id_to_dialect_Mid_Southern_1(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['Mid_Southern_1']

def convet_member_id_to_dialect_Lowland_Southern(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['Lowland_Southern']


def convet_member_id_to_dialect_22_New_English_2(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['New_English_2']


def convet_member_id_to_dialect_23_Voicing_Effect_column(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['Voicing_Effect']


def convet_member_id_to_dialect_24_Oregon_column(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['Oregon']


def convet_member_id_to_dialect_25_Oklahoma_column(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['Oklahoma']


def convet_member_id_to_dialect_26_New_Hampshire_column(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['New_Hampshire']


def convet_member_id_to_dialect_27_Washington_column(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['Washington']


def convet_member_id_to_dialect_28_Illinois_column(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['Illinois']


def convet_member_id_to_dialect_29_Michigan_column(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['Michigan']


def convet_member_id_to_dialect_30_New_Jersey_column(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['New_Jersey']


def convet_member_id_to_dialect_31_Washington_D_C_column(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['Washington_D_C']


def convet_member_id_to_dialect_majority_aw_fronting_01_column(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['majority_aw_fronting_F2_is_above_1750_Hz_red']

def convet_member_id_to_dialect_majority_aw_fronting_02_column(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['majority_aw_fronting_F2_is_less_than_1750_Hz_above_1650_Hz_yellow']

def convet_member_id_to_dialect_majority_aw_fronting_03_column(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['majority_aw_fronting_F2_is_less_than_1650_Hz_above_1450_Hz_green']

def convet_member_id_to_dialect_majority_aw_fronting_04_column(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['majority_aw_fronting_F2_is_less_than_1450_Hz_above_1200_Hz_blue']

def convet_member_id_to_dialect_majority_aw_fronting_05_column(state, df_dialects):
    state_row = df_dialects.loc[df_dialects['State name'] == state]
    return state_row.iloc[0]['majority_aw_fronting_F2_is_NaN']

def convet_member_id_to_us_state(state, df_dialects):
    return state


def get_soil_type_row_idx(row):
    if row[14] == 1:
        return 0
    if row[15] == 1:
        return 1
    if row[16] == 1:
        return 2
    if row[17] == 1:
        return 3
    if row[18] == 1:
        return 4
    if row[19] == 1:
        return 5
    if row[20] == 1:
        return 6
    if row[21] == 1:
        return 7
    if row[22] == 1:
        return 8
    if row[23] == 1:
        return 9
    if row[24] == 1:
        return 10
    if row[25] == 1:
        return 11
    if row[26] == 1:
        return 12
    if row[27] == 1:
        return 13
    if row[28] == 1:
        return 14
    if row[29] == 1:
        return 15
    if row[30] == 1:
        return 16
    if row[31] == 1:
        return 17
    if row[32] == 1:
        return 18
    if row[33] == 1:
        return 19
    if row[34] == 1:
        return 20
    if row[35] == 1:
        return 21
    if row[36] == 1:
        return 22
    if row[37] == 1:
        return 23
    if row[38] == 1:
        return 24
    if row[39] == 1:
        return 25
    if row[40] == 1:
        return 26
    if row[41] == 1:
        return 27
    if row[42] == 1:
        return 28
    if row[43] == 1:
        return 29
    if row[44] == 1:
        return 30
    if row[45] == 1:
        return 31
    if row[46] == 1:
        return 32
    if row[47] == 1:
        return 33
    if row[48] == 1:
        return 34
    if row[49] == 1:
        return 35
    if row[50] == 1:
        return 36
    if row[51] == 1:
        return 37
    if row[52] == 1:
        return 38
    if row[53] == 1:
        return 39


def convet_soil_type_to_water_storage(new_soil_features_idx, df_soil_features):

    Water_Storage_val = df_soil_features['Available Water Storage'].values[new_soil_features_idx]

    return Water_Storage_val


def convet_soil_type_to_water_percentage(new_soil_features_idx, df_soil_features):
    water_percentage_val = df_soil_features['water source percentage'].values[new_soil_features_idx]

    return water_percentage_val


def convet_soil_type_to_dominant_geomorphic_position(new_soil_features_idx, df_soil_features):
    dominant_geomorphic_position = df_soil_features['Dominant geomorphic Position'].values[new_soil_features_idx]
    return dominant_geomorphic_position



def convet_soil_type_to_rubble_land_percentage(new_soil_features_idx, df_soil_features):
    rubble_land_percentage = df_soil_features['Rubble land percentage'].values[new_soil_features_idx]
    return rubble_land_percentage



def convet_soil_type_to_rock_outcrop_percentage(new_soil_features_idx, df_soil_features):
    rock_outcrop_percentage = df_soil_features['rock outcrop percentage'].values[new_soil_features_idx]
    return rock_outcrop_percentage



def convet_soil_type_to_total_plant_available_water_1st(new_soil_features_idx, df_soil_features):
    majority_soil_family_percentage_1st = df_soil_features['majority_1 soil family percentage'].values[
                                              new_soil_features_idx] / 100

    total_plant_available_water_1st = majority_soil_family_percentage_1st * \
                                      df_soil_features['1st Total Plant Available Water'].values[new_soil_features_idx]
    return total_plant_available_water_1st



def convet_soil_type_to_organic_carbon_stock_0_100_cm_low_1st(new_soil_features_idx,
                                                              df_soil_features):
    majority_soil_family_percentage_1st = df_soil_features['majority_1 soil family percentage'].values[
                                              new_soil_features_idx] / 100
    organic_carbon_stock_0_100_cm_low_1st = majority_soil_family_percentage_1st * \
                                            df_soil_features['1st Organic Carbon Stock 0-100 cm kg-m2 low'].values[
                                                new_soil_features_idx]
    return organic_carbon_stock_0_100_cm_low_1st



def convet_soil_type_to_organic_carbon_stock_0_100_cm_high_1st(new_soil_features_idx,
                                                               df_soil_features):
    majority_soil_family_percentage_1st = df_soil_features['majority_1 soil family percentage'].values[
                                              new_soil_features_idx] / 100
    organic_carbon_stock_0_100_cm_high_1st = majority_soil_family_percentage_1st * \
                                             df_soil_features['1st Organic Carbon Stock 0-100 cm kg-m2 high'].values[
                                                 new_soil_features_idx]
    return organic_carbon_stock_0_100_cm_high_1st



def convet_soil_type_to_T_erosion_factor_1st(new_soil_features_idx, df_soil_features):
    majority_soil_family_percentage_1st = df_soil_features['majority_1 soil family percentage'].values[
                                              new_soil_features_idx] / 100
    T_erosion_factor_1st = majority_soil_family_percentage_1st * df_soil_features['1st T Erosion Factor'].values[
        new_soil_features_idx]
    return T_erosion_factor_1st


def convet_soil_type_to_total_plant_available_water_2nd(new_soil_features_idx, df_soil_features):
    majority_soil_family_percentage_2nd = df_soil_features['2nd majority soil family percentage'].values[
                                              new_soil_features_idx] / 100
    total_plant_available_water_2nd = majority_soil_family_percentage_2nd * \
                                      df_soil_features['2nd Total Plant Available Water'].values[new_soil_features_idx]
    return total_plant_available_water_2nd



def convet_soil_type_to_organic_carbon_stock_0_100_cm_low_2nd(new_soil_features_idx,
                                                              df_soil_features):
    majority_soil_family_percentage_2nd = df_soil_features['2nd majority soil family percentage'].values[
                                              new_soil_features_idx] / 100
    organic_carbon_stock_0_100_cm_low_2nd = majority_soil_family_percentage_2nd * \
                                            df_soil_features['2nd Organic Carbon Stock 0-100 cm kg-m2 low'].values[
                                                new_soil_features_idx]
    return organic_carbon_stock_0_100_cm_low_2nd



def convet_soil_type_to_organic_carbon_stock_0_100_cm_high_2nd(new_soil_features_idx,
                                                               df_soil_features):
    majority_soil_family_percentage_2nd = df_soil_features['2nd majority soil family percentage'].values[
                                              new_soil_features_idx] / 100
    organic_carbon_stock_0_100_cm_high_2nd = majority_soil_family_percentage_2nd * \
                                             df_soil_features['2nd Organic Carbon Stock 0-100 cm kg-m2 high'].values[
                                                 new_soil_features_idx]
    return organic_carbon_stock_0_100_cm_high_2nd



def convet_soil_type_to_T_erosion_factor_2nd(new_soil_features_idx, df_soil_features):
    majority_soil_family_percentage_2nd = df_soil_features['2nd majority soil family percentage'].values[
                                              new_soil_features_idx] / 100
    T_erosion_factor_2nd = majority_soil_family_percentage_2nd * df_soil_features['2nd T Erosion Factor'].values[
        new_soil_features_idx]
    return T_erosion_factor_2nd

if __name__ == '__main__':


    now = datetime.now()  # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    print("date and time:", date_time)

    covtype_flag = False
    isolet_flag = True


    choose_column_indices_with_soil = [0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                                       29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                       49, 50, 51, 52, 53, 54]


    if covtype_flag:
        run_on_dataset_covertype(choose_column_indices_with_soil)
    elif isolet_flag:
        # for orig_colum in range(300, 400):
        #     print("-----------------------------------------\n\n")
        #     print(f"isolet_flag: for orig_colum= {orig_colum}")
        #     run_on_dataset_isolet([orig_colum, 617]) ##
        print("FIRST_TIME:")
        run_on_dataset_isolet([37, 617])
        run_on_dataset_isolet([102, 617])
        print("SAME FEATURES -  2ND TIME:")
        run_on_dataset_isolet([37, 617])
        run_on_dataset_isolet([102, 617])



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
