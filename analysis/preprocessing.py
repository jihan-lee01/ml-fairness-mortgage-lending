import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(filepath="../data/2022_public_lar.csv"):
    return pd.read_csv(filepath)


def recode_categorical_cols(df, recode_map):
    for col, mapping in recode_map.items():
        df[col] = df[col].replace(mapping)
    return df


def preprocess(df):
    # Drop unnecessary, redundant, and sparse columns (based on domain knowledge)
    unnecessary = df.columns[list(
        range(47, 77)) + list(range(78, 99)) + [0, 1, 13, 18]]
    redundant = ['census_tract', 'derived_msa_md', 'county_code',
                 'loan_type', 'lien_status', 'construction_method']
    sparse = ['discount_points', 'total_points_and_fees', 'lender_credits',
              'prepayment_penalty_term', 'intro_rate_period', 'multifamily_affordable_units']
    df = df.drop(columns=unnecessary)
    df = df.drop(columns=redundant + sparse)

    # Drop observations with not useful values (e.g., Not Available, Free From Text Only)
    df = df[~df['derived_ethnicity'].isin(
        ['Ethnicity Not Available', 'Free Form Text Only', 'Joint'])]
    df = df[~df['derived_race'].isin(
        ['Race Not Available', 'Free Form Text Only', 'Joint'])]
    df = df[~df['derived_sex'].isin(['Sex Not Available', 'Joint'])]
    df = df[df['applicant_age'] != '8888']

    # Recode the target variable
    df = df[df['action_taken'].isin([1, 3])]
    df['loan_approved'] = df['action_taken'].apply(
        lambda x: 1 if x == 1 else 0)
    df = df.drop(columns=['action_taken'])

    # Rename certain columns and typecasting
    rename_map = {'derived_loan_product_type': 'loan_product_type', 'derived_dwelling_category': 'dwelling_category',
                  'derived_ethnicity': 'ethnicity', 'derived_race': 'race', 'derived_sex': 'sex', 'applicant_age': 'age'}
    df = df.rename(columns=rename_map).astype()

    # Change data types of some columns to numeric
    numeric_cols = ['combined_loan_to_value_ratio', 'interest_rate', 'rate_spread',
                    'origination_charges', 'loan_term', 'property_value', 'total_units', 'total_loan_costs']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

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

    X = df.drop('loan_approved', axis=1)
    y = df['loan_approved']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=334)

    # Recode protected attributes to Privileged: 1, Unprivileged: 0
    df['race'] = df['race'].apply(lambda x: 1 if x == 'White' else 0)
    df['ethnicity'] = df['ethnicity'].apply(
        lambda x: 1 if x == 'Not Hispanic or Latino' else 0)
    df['sex'] = df['sex'].apply(lambda x: 1 if x == 'Male' else 0)
    df['age'] = df['age'].apply(lambda x: 0 if x in ['<25', '>74'] else 1)

    # One-hot encoding
    categorical_cols = list(range(0, 4)) + [8, 13, 18, 19, 22]
    encoded = pd.get_dummies(df.iloc[:, categorical_cols])
    df = df.reset_index(drop=True)
    encoded = encoded.reset_index(drop=True)
    df = pd.concat([df, encoded], axis=1)
    df = df.drop(df.columns[categorical_cols], axis=1)

    return X_train, X_test, y_train, y_test


def imputing(X_train, X_test):
    # Impute missing values with median for floating point columns and most frequent values (mode) for others
    imputer_mode = SimpleImputer(strategy='most_frequent')
    imputer_median = SimpleImputer(strategy='median')
    for col in X_train.columns:
        if X_train[col].dtype == 'float64':
            X_train[col] = imputer_median.fit_transform(X_train[[col]]).ravel()
            X_test[col] = imputer_median.transform(X_test[[col]]).ravel()
        else:
            X_train[col] = imputer_mode.fit_transform(X_train[[col]]).ravel()
            X_test[col] = imputer_mode.transform(X_test[col]).ravel()

    return X_train, X_test


def scaling(X_train, X_test):
    scaler = StandardScaler()
    numeric_cols = ['loan_amount', 'combined_loan_to_value_ratio', 'interest_rate', 'rate_spread',
                    'total_loan_costs', 'origination_charges', 'loan_term', 'property_value', 'total_units', 'income']
    for col in numeric_cols:
        X_train[col] = scaler.fit_transform(X_train[col])
        X_test[col] = scaler.transform(X_test[col])

    return X_train, X_test


def encoding(X_train, X_test):
    categorical_cols = list(range(0, 4)) + [8, 13, 18, 19, 22]
    categorical_names = X_train.columns[categorical_cols]

    encoder = OneHotEncoder(handle_unknown='ignore')
    X_train_encoded = encoder.fit_transform(X_train.iloc[:, categorical_cols])
    X_test_encoded = encoder.transform(X_test.iloc[:, categorical_cols])

    new_col_names = encoder.get_feature_names_out(categorical_names)
    X_train_encoded = pd.DataFrame(
        X_train_encoded, index=X_train.index, columns=new_col_names)
    X_test_encoded = pd.DataFrame(
        X_test_encoded, index=X_test.index, columns=new_col_names)

    X_train = X_train.drop(columns=X_train.columns[categorical_cols])
    X_test = X_test.drop(columns=X_test.columns[categorical_cols])

    X_train = pd.concat([X_train, X_train_encoded], axis=1)
    X_test = pd.concat([X_test, X_test_encoded], axis=1)

    return X_train, X_test


def select_features(X_train, X_test, y_train):
    train_data = pd.concat([X_train, y_train], axis=1)
    corr_matrix = train_data.corr()
    target_corr = corr_matrix.iloc[:-1, -1].abs()
    col_to_drop = set()

    for i in range(len(corr_matrix.columns) - 1):
        for j in range(i+1, len(corr_matrix.columns) - 1):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                if target_corr[i] > target_corr[j]:
                    col_to_drop.add(corr_matrix.columns[j])
                else:
                    col_to_drop.add(corr_matrix.columns[i])

    X_train = X_train.drop(columns=list(col_to_drop))
    X_test = X_test.drop(columns=list(col_to_drop))

    return X_train, X_test


def main():
    print("-----Loading Data-----")
    df = load_data()
    print("----Preprocessing-----")
    X, y = preprocess(df)
    print("Preprocessing Complete.")


if __name__ == "__main__":
    main()
