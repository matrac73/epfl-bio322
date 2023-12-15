import pandas as pd
from rdkit import Chem
import numpy as np
import pytest
from module import (
    numerical_separator,
    standardizer,
    constant_predictors_remover,
    correlation_remover,
    extract_features_from_mol,
    mean_RT_for_duplicates,
    lab_bias,
    unbiased_RT,
    enrich,
    correlation_matrix,
    plot_correlation_matrix,
    features_reduction_using_correlation,
    choose_pca_percentage,
    PrincipalComponentAnalysis,
)

# Sample data for testing
sample_data = pd.DataFrame({
    'mol': [Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('CCO'),Chem.MolFromSmiles('CCCO'),Chem.MolFromSmiles('CCCCO')],
    'Compound': ['A', 'A', 'B', 'C'],
    'RT': [10.0, 11.0, 9, 5],
    'Lab': ['Lab1', 'Lab2', 'Lab2','Lab3'],
    'OtherColumn': [1, 2, 3, 4]
})

def test_extract_features_from_mol():
    result = extract_features_from_mol(sample_data)
    assert 'mol' not in result.columns
    assert 'MolecularWeight' in result.columns
    assert 'TotalAtoms' in result.columns

def test_mean_RT_for_duplicates():
    result = mean_RT_for_duplicates(sample_data)
    assert 'mean_RT' in result.columns

def test_lab_bias():
    result = lab_bias(sample_data)
    assert 'Bias' in result.columns
    assert 'lab_mean_bias' in result.columns

def test_unbiased_RT():
    result = unbiased_RT(sample_data)
    assert 'Corrected_RT' in result.columns

def test_correlation_matrix():
    result = correlation_matrix(sample_data)
    assert isinstance(result, pd.DataFrame)

def test_features_reduction_using_correlation():
    correlation_matrix_data = sample_data.corr()
    threshold = 0.7
    result = features_reduction_using_correlation(correlation_matrix_data, threshold, sample_data)
    assert result.shape[1] < sample_data.shape[1]

def test_numerical_separator():
    non_numerical, numerical = numerical_separator(sample_data)
    assert isinstance(non_numerical, pd.DataFrame)
    assert isinstance(numerical, pd.DataFrame)

def test_standardizer():
    result = standardizer(sample_data)
    assert 'OtherColumn' in result.columns
    assert 'Standardized_RT' in result.columns

def test_choose_pca_percentage():
    with pytest.raises(SystemExit):
        choose_pca_percentage(sample_data)

def test_PrincipalComponentAnalysis():
    result = PrincipalComponentAnalysis(sample_data, 0.95)
    assert 'PC1' in result.columns
    assert 'Lab' in result.columns
