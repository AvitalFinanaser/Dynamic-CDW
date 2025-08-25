import pandas as pd
import os


def get_project_root():
    """Get the absolute path to the project root directory"""
    # This assumes demographic_utils.py is in the 'datasets' directory
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # Get parent directory (project root)
    project_root = os.path.dirname(current_file_dir)
    return project_root


def observe_demographic_values():
    # Load dataset
    project_root = get_project_root()
    csv_path = os.path.join(project_root, 'datasets', 'UNdata_2022_Israel.csv')
    df = pd.read_csv(csv_path)

    # Dataset observation
    for feature in df.columns:
        if feature != 'Proportion':
            uni = df[feature].unique()
            print(f"{feature}: {uni}")


def prepare_demographic_data() -> pd.DataFrame:
    # Load dataset
    project_root = get_project_root()
    csv_path = os.path.join(project_root, 'datasets', 'UNdata_2022_Israel.csv')
    df = pd.read_csv(csv_path)

    # 1. Dataset preprocessing

    # Remove "ISCED 2011, Total" in Educational attainment
    df = df[df['Educational attainment'] != 'ISCED 2011, Total']

    # Remove "Both" in Sex
    df = df[df['Sex'] != 'Both Sexes'].copy()

    # 2. Normalize 'Value' column to proportions
    total_value = df['Value'].sum()
    df['Proportion'] = df['Value'] / total_value

    # 3. Keep only relevant columns
    df_prepared = df[['Sex', 'Age', 'Educational attainment', 'Proportion']]

    return df_prepared


def sample_profiles(df_prepared: pd.DataFrame, num_samples: int):
    """
    Sample agent profiles from the demographic data with weighted random sampling.

    Args:
        df_prepared: DataFrame with normalized proportions.
        num_samples: Number of profiles to sample.

    Returns:
        List of profile dictionaries.
    """
    sampled_rows = df_prepared.sample(
        n=num_samples,
        weights='Proportion',
        replace=True  # Allows duplicates
    ).reset_index(drop=True)

    profiles = sampled_rows.apply(
        lambda row: {
            'Sex': row['Sex'],
            'Age': row['Age'],
            'Educational attainment': row['Educational attainment']
        }, axis=1
    ).tolist()

    return profiles


def verify_sample_profiles(sampled_profiles: pd.DataFrame, df_prepared: pd.DataFrame) -> pd.DataFrame:
    """
    Check Distribution of Sampled Profiles:
    - Count frequency of each profile in your sampled list.
    - Compare frequencies vs. expected proportions.
    Args:
        df_prepared: DataFrame with normalized proportions.
        sampled_profiles: DataFrame with sampled profiles.
    Returns:
       Comparison Dataframe.
    """
    # Convert to DataFrame for analysis
    sampled_df = pd.DataFrame(sampled_profiles)
    # Group by profile combinations and count
    sampled_counts = sampled_df.groupby(['Sex', 'Age', 'Educational attainment']).size().reset_index(name='Count')
    # Merge with original proportions for comparison
    df_merged = pd.merge(
        sampled_counts,
        df_prepared,
        on=['Sex', 'Age', 'Educational attainment'],
        how='left'
    )

    # Calculate sampled proportion
    df_merged['Sampled Proportion'] = df_merged['Count'] / df_merged['Count'].sum()

    # Compare with original
    df_merged['Difference'] = df_merged['Sampled Proportion'] - df_merged['Proportion']

    # Display comparison
    return df_merged[['Sex', 'Age', 'Educational attainment', 'Proportion', 'Sampled Proportion', 'Difference']]


def test_utilis():
    observe_demographic_values()
    df_prepared = prepare_demographic_data()
    # Save to JSON
    save_path = os.path.join('datasets', 'processed', 'prepared_demographics.json')
    df_prepared.to_json(save_path, orient='records', indent=4)
    # Sample several profiles
    sampled_profiles = sample_profiles(df_prepared, num_samples=1000)
    # Verify sample method
    diff = verify_sample_profiles(sampled_profiles=sampled_profiles, df_prepared=df_prepared)
    print(f"Difference average: {diff['Difference']}")
