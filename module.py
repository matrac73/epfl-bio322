import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA




def numerical_separator(df): 
    """
    Separate the numerical columns for the other in a Dataframe. 
    
    Parameters: 
    - df (pd.dataframe) : Dataframe containing numerical and non numerical columns 
    
    Return: 
    - colonnes_non_numeriques : non numerical columns from the Original Dataframe 
    - colonnes_numeriques : numerical colums form the Original Dataframe
    """
    colonnes_numeriques = df.select_dtypes(include=['number'])
    colonnes_non_numeriques = df.select_dtypes(exclude=['number'])
    return  colonnes_non_numeriques, colonnes_numeriques


# CLEANING

def standardizer(df, target='RT'):
    """
    Standardize numerical features in a DataFrame, including the target feature if present.

    Parameters:
    - df (pd.DataFrame): DataFrame to standardize.
    - target (str): Name of the target feature (default is 'RT').

    Returns:
    - pd.DataFrame: Original DataFrame with standardized numerical features.
    """
    non_numerical_columns, numerical_columns = numerical_separator(df)

    # Extract the target feature, if present
    if target in numerical_columns.columns:
        target_feature = numerical_columns[target]
        numerical_columns = numerical_columns.drop(columns=[target])

        # Standardize numerical features
        scaler = StandardScaler()
        scaled_numerical_data = pd.DataFrame(scaler.fit_transform(numerical_columns),
                                             columns=numerical_columns.columns)

        # Concatenate non-numerical columns, target feature, and scaled numerical data
        non_numerical_columns.reset_index(drop=True, inplace=True)
        scaled_numerical_data.reset_index(drop=True, inplace=True)
        df_scaled = pd.concat([non_numerical_columns, target_feature, scaled_numerical_data], axis=1)

    else:
        # Standardize numerical features (excluding target feature)
        scaler = StandardScaler()
        scaled_numerical_data = pd.DataFrame(scaler.fit_transform(numerical_columns),
                                             columns=numerical_columns.columns)

        # Concatenate non-numerical columns and scaled numerical data
        non_numerical_columns.reset_index(drop=True, inplace=True)
        scaled_numerical_data.reset_index(drop=True, inplace=True)
        df_scaled = pd.concat([non_numerical_columns, scaled_numerical_data], axis=1)

    return df_scaled

def constant_predictors_remover(df) : 
    """
    Remove constant columns from a Dataframe. 
    
    Parameters : 
    - df (pd.dataframe) : Dataframe containing constant columns.
    
    Returns : 
    - pd.dtaframe : Original Dataframe without constant columns.
    """
    non_numerical_columns, numerical_columns = numerical_separator(df) #splitting the non numerical and numerical columns in data frame
    std_per_column = np.std(numerical_columns, axis=0) #calculating std for each numerical column 
    non_constant_columns = std_per_column[std_per_column != 0].index #selectioning non constant columns
    df_clean_const_numerical = df.loc[:, non_constant_columns]
    df_clean_const = pd.concat([non_numerical_columns, df_clean_const_numerical], axis=1)
    return df_clean_const

def correlation_remover(df) : 
    """
    Remove the correlated columns in a Dataframe. 
    
    Parameters : 
    - df (pd.dataframe) : Dataframe with some perfectly correlated columns. 
    
    Retruns : 
    - pd.dataframe : Original Dataframe with only non correlated columns (one of each group of prefectly correlated columns remains).
    """
    non_numerical_columns, numerical_columns = numerical_separator(df)
    correlation = np.array(numerical_columns.corr().values)
    correlation = np.triu(correlation, k=0)
    np.fill_diagonal(correlation,0)
    df_clean_corr_numerical = numerical_columns.drop(numerical_columns.columns[np.where(correlation==1)[1]], axis=1)
    df_clean_corr = pd.concat([non_numerical_columns, df_clean_corr_numerical], axis=1)
    return df_clean_corr

def NaN_checker(df, pt=False) : 
    """
    Check for the presence of NaN in a Dataframe.
    
    Parameters: 
    - df (pd.dataframe): Dataframe potentially containing NaN.
    
    Returns : 
    - nan_check : boolean value indicating if there is NaN values in the Dataframe. 
    """
    nan_check = df.isna().any().any()
    if pt == True :
        print("Il y a des NaN dans le DataFrame :", nan_check)
    return nan_check
    
def NaN_remover(df): 
    """
    Remove the NaN values from a Dataframe. 
    
    Parameters : 
    - df (pd.dataframe): Dataframe containing NaN.
    
    Returns : 
    - pd.dataframe : Original Dataframe without the lines containing NaN values 
    """
    df_without_NaN = df.copy()
    df_without_NaN.dropna(inplace=True)
    return df_without_NaN

#only use :
def clean(data):
    data = standardizer(data)
    data = constant_predictors_remover(data)
    data = correlation_remover(data)
    data = NaN_remover(data)
    return data

# FEATURE ENRICHMENT functions 

def extract_features_from_mol(df):
    """
    Extract molecular features from a column of molecules in a DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing a column named 'mol' with RDKit molecule objects.

    Returns:
    - pd.DataFrame: Original DataFrame with additional columns for extracted molecular features.
    """
    df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    
    # Initialize an empty DataFrame
    mol_features_df = pd.DataFrame()

    # Loop through each molecule in the 'mol' column
    for molecule in df['mol']:
        # Calculate features 
        mol_weight = Descriptors.MolWt(molecule)
        total_atoms = molecule.GetNumAtoms()
        carbon_atoms = Descriptors.HeavyAtomCount(molecule)
        num_rings = Descriptors.RingCount(molecule)
        rotatable_bonds = Descriptors.NumRotatableBonds(molecule)
        tpsa = Descriptors.TPSA(molecule)
        mol_log_p = Descriptors.MolLogP(molecule)
        num_h_acceptors = Lipinski.NumHAcceptors(molecule)
        num_h_donors = Lipinski.NumHDonors(molecule)
        num_valence_electrons = Descriptors.NumValenceElectrons(molecule)
        num_aliphatic_carbocycles = Lipinski.NumAliphaticCarbocycles(molecule)
        num_aliphatic_heterocycles = Lipinski.NumAliphaticHeterocycles(molecule)

        # Append features to a temporary DataFrame
        temp_df = pd.DataFrame({
            'MolecularWeight': [mol_weight],
            'TotalAtoms': [total_atoms],
            'CarbonAtoms': [carbon_atoms],
            'NumRings': [num_rings],
            'RotatableBonds': [rotatable_bonds],
            'TPSA': [tpsa],
            'MolLogP': [mol_log_p],
            'NumHAcceptors': [num_h_acceptors],
            'NumHDonors': [num_h_donors],
            'NumValenceElectrons': [num_valence_electrons],
            'NumAliphaticCarbocycles': [num_aliphatic_carbocycles],
            'NumAliphaticHeterocycles': [num_aliphatic_heterocycles]    
        })

        # Concatenate temporary DataFrame to the main DataFrame
        mol_features_df = pd.concat([mol_features_df, temp_df], ignore_index=True)

    # Concatenate molecular features to the original DataFrame
    df = pd.concat([df, mol_features_df], axis=1)
    df.drop('mol', axis=1, inplace=True)
    
    return df

def mean_RT_for_duplicates(data):
    """
    Calculate the mean RT for compounds with duplicates and add a 'mean_RT' column to the DataFrame.

    Parameters:
    - data (pandas.DataFrame): Input DataFrame containing 'Compound' and 'RT' columns.

    Returns:
    - pandas.DataFrame: DataFrame with an additional 'mean_RT' column.
    """
    # Compute mean RT for each duplicate group
    mean_RT_values = data.groupby('Compound')['RT'].transform('mean')

    # Add mean_RT column to the DataFrame
    data['mean_RT'] = mean_RT_values

    return data

def lab_bias(data) :#inutile
    """
    Calculate lab-specific biases in retention time.

    This function calculates lab-specific biases by first computing the mean
    retention time for compounds with duplicates in the provided dataset.
    It then calculates the lab-specific bias for each data point by subtracting
    the mean retention time from the actual retention time. Finally, the function
    computes the lab-specific mean bias and adds a 'lab_mean_bias' column to the DataFrame.

    Parameters:
    - data (pandas.DataFrame): Input DataFrame containing 'RT', 'Lab', and other relevant columns.

    Returns:
    - pandas.DataFrame: DataFrame with additional columns for lab-specific biases.
    """
    #prerequisite dataset treatment
    mean_RT_for_duplicates(data)
    
    #Calculate Lab-Specific Bias
    data['Bias'] = data['RT'] - data['mean_RT']
    
    #Calculate Lab-Specific Mean Bias
    mean_bias = data.groupby('Lab')['Bias'].transform('mean')
    
    # Add lab_mean_bias column to the DataFrame
    data['lab_mean_bias'] = mean_bias
    
    return data


# only use 
# A MODIFIER  ADD CDDD MERGE
# ET A COPIER COLLER DANS LE NOTEBOOK FEATURE ENRICHMENT AU NIVEAU DE 'ENRICHMEN FUNCTION '
def enrich(data): #only for training data 
    """Enriches the training data by developping the feature mol, merging cddd and standardizing the new enriching features

    Args:
        data (panda.DataFrame): imput data

    Returns:
        panda.DataFrame: enriched dataframe 
    """
    enriched_Data = extract_features_from_mol(data)
    cddd = pd.read_csv('cddd.csv')
    enriched_Data_cd = pd.merge(enriched_Data, cddd, on='SMILES', how='left')
    if NaN_checker(enriched_Data_cd) == True : 
         nned, ned = numerical_separator(enriched_Data_cd)
         moyennes_colonnes = ned.mean()
         ned = ned.fillna(moyennes_colonnes)
         nned.reset_index(drop=True, inplace=True)
         ned.reset_index(drop=True, inplace=True)
         enriched_Data_wn = pd.concat([nned, ned], axis=1)
    else : 
        enriched_Data_wn = enriched_Data_cd
    enriched_Data_st = standardizer(enriched_Data_wn)
    return enriched_Data_st





# VISUALISATION
def plot_histogram(data, feature='RT'):
    """
    Plot a histogram of the distribution of a feature in a DataFrame.

    Parameters:
    - data: pd.DataFrame, the input dataset
    - feature: str, the name of the feature to plot (default is 'RT')
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data[feature], bins=30, kde=True, color='blue', edgecolor='black')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

def correlation_matrix (data):
    """
    Calculate the correlation matrix for numeric columns in a DataFrame.

    Parameters:
    - data (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: Correlation matrix for numeric columns.
    """
    # non-numeric columns are excluded (we only use the ones converted to numeric)
    numeric_df = data.select_dtypes(include=['float64', 'int64'])

    # Calculate the correlation matrix
    correlation_matrix = numeric_df.corr()
    
    return correlation_matrix

def plot_correlation_matrix (data) :
    """
    Plot the correlation matrix using a heatmap.

    Parameters:
    - matrix (pd.DataFrame): Correlation matrix.

    Returns:
    - None
    """
    # Plot the correlation matrix using seaborn
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix(data), annot=False, cmap='coolwarm',vmin=-1, vmax=1, center=0)
    plt.title('Correlation Matrix')
    plt.show()
    return None


# FEATURES REDUCTION
def features_reduction_using_correlation(correlation_matrix, threshold, dataset):
    """
    Reduce features in a dataset based on a correlation matrix.

    Parameters:
    - correlation_matrix (pd.DataFrame): Correlation matrix of the dataset.
    - threshold (float): Threshold for correlation. Features with correlation above this value will be dropped.
    - dataset (pd.DataFrame): Original dataset.

    Returns:
    - pd.DataFrame: Dataset with reduced features.
    """
    # Create a mask to identify highly correlated features
    upper_triangle = np.triu(np.ones(correlation_matrix.shape), k=1)
    mask = (correlation_matrix.abs() > threshold) & upper_triangle

    # Identify features to drop
    to_drop = set()
    for column in mask.columns:
        correlated_columns = mask.index[mask[column]]
        to_drop.update(set(correlated_columns) - {column})

    # Drop highly correlated features from the dataset
    dataset_filtered = dataset.drop(columns=to_drop)
    
    return dataset_filtered

def choose_pca_percentage(data, target='Corrected_RT', threshold=0.90, step=0.05):
    """
    Choose the optimal percentage of variance to retain for PCA based on cumulative explained variance.

    Parameters:
    - data: pandas DataFrame, the input dataset
    - target: str, the name of the target variable column
    - threshold: float, the desired threshold for cumulative explained variance
    - step: float, the step size for trying different percentages

    Returns:
    - optimal_percentage: float, the chosen percentage of variance to retain
    """
    # Separate numerical and non-numerical columns
    non_numerical_columns, numerical_columns = numerical_separator(data)

    # Standardize numerical features
    standardized_data = standardizer(data)

    # Apply PCA without specifying the number of components
    pca = PCA()
    X_pca = pca.fit_transform(standardized_data[numerical_columns.columns])

    # Explained variance ratio and cumulative explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    # Visualize the cumulative explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--', label='Cumulative Explained Variance')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance for Different Numbers of Components')
    plt.legend()
    plt.show()

    # Choose the optimal percentage based on the threshold
    optimal_percentage = np.argmax(cumulative_explained_variance >= threshold) * step

    print(f'Optimal Percentage: {optimal_percentage}%')
    return optimal_percentage

def PrincipalComponentAnalysis(data, keeping_percentage, target='RT'):
    """
    Perform Principal Component Analysis (PCA) on a given dataset.

    Parameters:
    - data: pandas DataFrame, the input dataset
    - keeping_percentage: float, the desired percentage of variance to retain
    - target: str, the name of the target variable column

    Returns:
    - result: pandas DataFrame, the dataset with reduced features after PCA
    """
    # Separate numerical and non-numerical columns
    non_numerical_columns, numerical_columns = numerical_separator(data)

    # Standardize numerical features
    standardized_data = standardizer(data) #NUMERICAL?

    # Apply PCA with the specified percentage of variance to retain
    pca = PCA(keeping_percentage)
    X_pca = pca.fit_transform(standardized_data[numerical_columns.columns])

    # Separate features and target variable
    X = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    y = data[target]

    # Concatenate the reduced features with the target variable and non-numerical columns
    result = pd.concat([non_numerical_columns, X, y], axis=1)

    return result

#PCA for both train and test 
def PCAtesttrain(trainingdata, testdata, n_components , target='RT'):
    
    # Separate numerical and non-numerical columns 
    non_numerical_columns_tr, numerical_columns_tr = numerical_separator(trainingdata)
    non_numerical_columns_te, numerical_columns_te = numerical_separator(testdata)
    # Standardize numerical features in train and test
    standardized_data_tr = standardizer(numerical_columns_tr)
    standardized_data_te = standardizer(numerical_columns_te)
    
    #merge standardized numerical training and test set 
    standardized_data = pd.concat([standardized_data_tr, standardized_data_te], ignore_index=True)#check if well done
    
    # Apply PCA to both test and train
    pca = PCA(n_components)
    X_pca_traintest = pca.fit_transform(standardized_data)
    
    #SEPERATE TRAIN AND TEST 
    
    #ADD BACK TARGET TO TRAIN 
    y = trainingdata[target]
    #ADD BACK NON NUMERICAL TO TRAIN AND TEST
    # Concatenate the reduced features with the target variable and non-numerical columns
    resultTRAIN = pd.concat([non_numerical_columns, X_pca, y], axis=1)
    RESULTTEST= 
    return resultTRAIN, RESULTTEST

# FINAL CORRECTION OF RT
def lab_bias_df(data) : #on trining
    """
    Calculate lab-specific biases in retention time.

    This function calculates lab-specific biases by first computing the mean
    retention time for compounds with duplicates in the provided dataset.
    It then calculates the lab-specific bias for each data point by subtracting
    the mean retention time from the actual retention time. 
    Finally, the function computes the lab-specific mean bias and 
    creates a new DataFrame containing every labs and their mean bias.

    Parameters:
    - data (pandas.DataFrame): Input (train) DataFrame containing 'RT', 'Lab', and other relevant columns.

    Returns:
    - pandas.DataFrame: DataFrame containing Labs and their lab-specific biases.
    """
    #prerequisite dataset treatment
    mean_RT_for_duplicates(data) #adds column mean rt to data
    
    #Calculate Lab-Specific Bias
    data['Bias'] = data['RT'] - data['mean_RT']
    
    # Calculate mean bias for each lab :pd serie
    mean_bias = data.groupby('Lab')['Bias'].mean().reset_index()

    # Create a new DataFrame with unique Lab values and their corresponding mean bias
    lab_bias_df = pd.DataFrame({
        'Lab': mean_bias['Lab'],
        'LabMeanBias': mean_bias['Bias']
    })
    
    return lab_bias_df

def unbiased_RT(RTdf, test, lab_bias): # to use on the final prediction (dataframe containing only the guessed rt)
    """
    Compute the corrected Retention Time (RT) without the lab bias. if the lab is unknown, the rt is not modified.

    Parameters:
    - RTdf (pandas.DataFrame): RT predictions from the test set (dataframe containing only the guessed rt)
    - test (pandas.DataFrame): test dataframe
    - lab_bias (pandas.DataFrame): DataFrame containing Labs and their lab_mean_bias (calculated from train)

    Returns:
    - pandas.DataFrame: new df containing only the corrected RT
    """
    # create a new column 'lab_bias' in test, that contains the lab's mean bias (found in 'lab_bias' df),
    # if the lab is unknown, consider its mean bias as 0
    test['lab_bias'] = RTdf['Lab'].map(lab_bias.set_index('Lab')['LabMeanBias']).fillna(0)
    test.head()
    # create new df 'ordered_lab_bias' only containing the 'lab bias' column of test
    ordered_lab_bias = test['lab_bias']
    ordered_lab_bias.head()
    # compute the corrected RT without the lab bias
    RTdf['Corrected_RT'] = RTdf['RT'] - ordered_lab_bias
    RTdf.head()
    RTdf = RTdf.drop('RT', axis=1)
    
    return RTdf
data = pd.read_csv('train.csv')
data['mol'] = data['SMILES'].apply(Chem.MolFromSmiles)
extract_features_from_mol(data)
mean_RT_for_duplicates(data)
lab_bias(data)
#unbiased_RT(data)





    





