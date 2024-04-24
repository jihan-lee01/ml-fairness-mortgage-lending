import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_data(filepath="../data/2022_public_lar.csv"):
    return pd.read_csv(filepath)


def drop_cols(df, cols):
    return df.drop(columns=cols)


def impute_numeric_cols(df, cols, strategy='median'):
    imputer = SimpleImputer(strategy=strategy)
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = imputer.fit_transform(df[[col]]).ravel()
    return df


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

    # Drop NAs
    df = df.dropna()

    # Drop observations with useless values (e.g., Not Available, Free From Text Only)
    df = df[(df['derived_ethnicity'] != 'Ethnicity Not Available')
            & (df['derived_ethnicity'] != 'Free Form Text Only')]
    df = df[(df['derived_race'] != 'Race Not Available') &
            (df['derived_race'] != 'Free Form Text Only')]
    df = df[df['derived_sex'] != 'Sex Not Available']
    df = df[df['applicant_age'] != '8888']

    # Recode the target variable
    df = df[df['action_taken'].isin([1, 2, 3, 7, 8])]
    df['loan_approved'] = df['action_taken'].apply(
        lambda x: 1 if x in [1, 2, 8] else 0)
    df = drop_cols(df, ['action_taken'])

    # Impute numeric columns
    numeric_cols = ['total_loan_costs', 'origination_charges', 'loan_term', 'property_value',
                    'combined_loan_to_value_ratio', 'interest_rate', 'rate_spread']
    df = impute_numeric_cols(df, numeric_cols)

    # Rename certain columns and typecasting
    rename_map = {'derived_msa_md': 'msa_md', 'derived_loan_product_type': 'loan_product_type', 'derived_dwelling_category': 'dwelling_category',
                  'derived_ethnicity': 'ethnicity', 'derived_race': 'race', 'derived_sex': 'sex', 'applicant_age': 'age'}
    df = df.rename(columns=rename_map).astype(
        {'msa_md': object, 'loan_term': int, 'total_units': int, 'loan_approved': object})

    # Replace values before one-hot encoding
    recode_map = {
        'ethnicity': {'Not Hispanic or Latino': 'Non-Hispanic', 'Hispanic or Latino': 'Hispanic'},
        'race': {'Black or African American': 'Black', 'American Indian or Alaska Native': 'Native',
                 'Native Hawaiian or Other Pacific Islander': 'PI', '2 or more minority races': '2+Minority'},
        'preapproval': {2: 0},
        'loan_purpose': {1: 'Purchase', 2: 'Improvement', 31: 'Refinancing', 32: 'Cash-out', 4: 'Other', 5: 'N/a'},
        'open_end_line_of_credit': {1: 'Yes', 2: 'No', 1111: 'Exempt'},
        'business_or_commercial_purpose': {1: 'Yes', 2: 'No', 1111: 'Exempt'},
        'hoepa_status': {1: 'Yes', 2: 'No', 3: 'N/a'},
        'negative_amortization': {1: 'Yes', 2: 'No', 1111: 'Exempt'},
        'interest_only_payment': {1: 'Yes', 2: 'No', 1111: 'Exempt'},
        'balloon_payment': {1: 'Yes', 2: 'No', 1111: 'Exempt'},
        'other_nonamortizing_features': {1: 'Yes', 2: 'No', 1111: 'Exempt'},
        'occupancy_type': {1: 'Principal', 2: 'Second', 3: 'Investment'},
        'manufactured_home_secured_property_type': {1: 'Home&Land', 2: 'Home', 3: 'N/a', 1111: 'Exempt'},
        'manufactured_home_land_property_interest': {1: 'Direct', 2: 'Indirect', 3: 'Paid', 4: 'Unpaid', 5: 'N/a', 1111: 'Exempt'},
        'debt_to_income_ratio': {39.0: '39', 38.0: '38', 37.0: '37'}
    }
    df = recode_categorical_cols(df, recode_map)
    df['preapproval'] = df['preapproval'].astype(object)

    # Re-scaling numeric columns
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['number'])
    df[numeric_cols.columns] = scaler.fit_transform(numeric_cols)

    # One-hot encoding
    categorical_cols = list(range(0, 7)) + list(range(8, 11)) + [15] + list(
        range(19, 23)) + list(range(24, 27)) + list(range(29, 31))
    encoded = pd.get_dummies(df.iloc[:, categorical_cols])
    df = df.reset_index(drop=True)
    encoded = encoded.reset_index(drop=True)
    df = pd.concat([df, encoded], axis=1)
    df = df.drop(df.columns[categorical_cols], axis=1)

    # Save preprocessed data
    df.to_csv("../data/preprocessed_data.csv", index=False)

    # Separate features and target
    y = df['loan_approved'].to_numpy().flatten()
    X = df.drop('loan_approved', axis=1).to_numpy()

    return X, y


def main():
    print("-----Loading Data-----")
    df = load_data()
    print("----Preprocessing-----")
    X, y = preprocess(df)
    print("Preprocessing Complete.")


if __name__ == "__main__":
    main()
