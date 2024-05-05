import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath="../data/2022_public_lar.csv"):
    return pd.read_csv(filepath)


def drop_cols(df, cols):
    return df.drop(columns=cols)


def recode_categorical_cols(df, recode_map):
    for col, mapping in recode_map.items():
        df[col] = df[col].replace(mapping)
    return df


def preprocess(df):
    # Drop unnecessary, redundant, and sparse columns (based on domain knowledge)
    unnecessary = df.columns[list(
        range(47, 77)) + list(range(78, 99)) + [0, 1, 13, 18]]
    redundant = ['census_tract', 'state_code', 'county_code',
                 'loan_type', 'lien_status', 'construction_method']
    sparse = ['discount_points', 'total_points_and_fees', 'lender_credits',
              'prepayment_penalty_term', 'intro_rate_period', 'multifamily_affordable_units']
    df = drop_cols(df, unnecessary)
    df = drop_cols(df, redundant)
    df = drop_cols(df, sparse)

    # Drop observations with not useful values (e.g., Not Available, Free From Text Only)
    df = df[(df['derived_ethnicity'] != 'Ethnicity Not Available') & (
        df['derived_ethnicity'] != 'Free Form Text Only') & (df['derived_ethnicity'] != 'Joint')]
    df = df[(df['derived_race'] != 'Race Not Available') & (
        df['derived_race'] != 'Free Form Text Only') & (df['derived_race'] != 'Joint')]
    df = df[(df['derived_sex'] != 'Sex Not Available')
            & (df['derived_sex'] != 'Joint')]
    df = df[df['applicant_age'] != '8888']

    # Recode the target variable
    df = df[df['action_taken'].isin([1, 3])]
    df['loan_approved'] = df['action_taken'].apply(
        lambda x: 1 if x == 1 else 0)
    df = drop_cols(df, ['action_taken'])

    # Change data types of some columns to numeric
    numeric_cols = ['combined_loan_to_value_ratio', 'interest_rate', 'rate_spread',
                    'origination_charges', 'loan_term', 'property_value', 'total_units', 'total_loan_costs']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Impute missing values with median for floating point columns and most frequent values (mode) for others
    imputer_mode = SimpleImputer(strategy='most_frequent')
    imputer_median = SimpleImputer(strategy='median')
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = imputer_median.fit_transform(df[[col]]).ravel()
        else:
            df[col] = imputer_mode.fit_transform(df[[col]]).ravel()

    # Rename certain columns and typecasting
    rename_map = {'derived_msa_md': 'msa_md', 'derived_loan_product_type': 'loan_product_type', 'derived_dwelling_category': 'dwelling_category',
                  'derived_ethnicity': 'ethnicity', 'derived_race': 'race', 'derived_sex': 'sex', 'applicant_age': 'age'}
    df = df.rename(columns=rename_map).astype(
        {'msa_md': object, 'loan_term': int, 'total_units': int})

    # Drop highly correlated features (preserve features which have higher correlation with the target)
    corr = df.corr()
    target_corr = corr.iloc[:-1, -1].abs()
    col_to_drop = set()

    for i in range(len(corr.columns) - 1):
        for j in range(i+1, len(corr.columns) - 1):
            if abs(corr.iloc[i, j]) > 0.9:
                if target_corr[i] > target_corr[j]:
                    col_to_drop.add(corr.columns[j])
                else:
                    col_to_drop.add(corr.columns[i])

    df = df.drop(columns=list(col_to_drop))

    df['loan_approved'] = df['loan_approved'].astype(object)

    # Recode protected attributes to Privileged: 1, Unprivileged: 0
    df['race'] = df['race'].apply(lambda x: 1 if x == 'White' else 0)
    df['ethnicity'] = df['ethnicity'].apply(
        lambda x: 1 if x == 'Not Hispanic or Latino' else 0)
    df['sex'] = df['sex'].apply(lambda x: 1 if x == 'Male' else 0)
    df['age'] = df['age'].apply(lambda x: 0 if x in ['<25', '>74'] else 1)

    # Replace values before one-hot encoding
    recode_map = {
        'preapproval': {2: 0},
        'loan_purpose': {1: 'Purchase', 2: 'Improvement', 31: 'Refinancing', 32: 'Cash-out', 4: 'Other', 5: 'N/a'},
        'hoepa_status': {1: 'Yes', 2: 'No', 3: 'N/a'},
        'occupancy_type': {1: 'Principal', 2: 'Second', 3: 'Investment'},
        'manufactured_home_land_property_interest': {1: 'Direct', 2: 'Indirect', 3: 'Paid', 4: 'Unpaid', 5: 'N/a', 1111: 'Exempt'},
        'debt_to_income_ratio': {'39.0': '39', '38.0': '38'}
    }
    df = recode_categorical_cols(df, recode_map)
    df['preapproval'] = df['preapproval'].astype(object)

    # Re-scaling
    scaler = StandardScaler()
    numeric_cols = ['loan_amount', 'combined_loan_to_value_ratio', 'interest_rate', 'rate_spread',
                    'total_loan_costs', 'origination_charges', 'loan_term', 'property_value', 'total_units', 'income']
    for col in numeric_cols:
        df[col] = scaler.fit_transform(df[[col]])

    # One-hot encoding
    df['msa_md'] = df['msa_md'].astype(object)
    categorical_cols = list(range(0, 4)) + [8, 13, 18, 19, 22]
    encoded = pd.get_dummies(df.iloc[:, categorical_cols])
    df = df.reset_index(drop=True)
    encoded = encoded.reset_index(drop=True)
    df = pd.concat([df, encoded], axis=1)
    df = df.drop(df.columns[categorical_cols], axis=1)

    # Save preprocessed data
    df.to_csv("../data/preprocessed_data.csv", index=False)

    # Separate features and target
    y = df['loan_approved']
    X = df.drop('loan_approved', axis=1)

    return X, y


def main():
    print("-----Loading Data-----")
    df = load_data()
    print("----Preprocessing-----")
    X, y = preprocess(df)
    print("Preprocessing Complete.")


if __name__ == "__main__":
    main()
