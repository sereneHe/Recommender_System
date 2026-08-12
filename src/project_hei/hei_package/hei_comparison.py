import pandas as pd


def compare_hei_columns(excel_file1, excel_file2):
    """
    Reads two Excel files, compares the 'HEI' column, and prints the differences.

    Args:
        excel_file1 (str): Path to the first Excel file.
        excel_file2 (str): Path to the second Excel file.
    """
    if excel_file1.endswith(".csv"):
        df1 = pd.read_csv(excel_file1)
    else:
        df1 = pd.read_excel(excel_file1)
    if excel_file2.endswith(".csv"):
        df2 = pd.read_csv(excel_file2)
    else:
        df2 = pd.read_excel(excel_file2)

    if "HEI" not in df1.columns or "HEI" not in df2.columns:
        print("Error: 'HEI' column not found in one or both Excel files.")
        return

    hei1 = df1["HEI"]
    hei2 = df2["HEI"]

    hei_diff = hei1 - hei2

    max_error = abs(hei_diff).max()
    mean_error = abs(hei_diff).mean()
    print(f"Maximum discrepancy: {max_error}\nMean discrepancy: {mean_error}")


if __name__ == "__main__":
    excel_file1_path = "data/my_HEI.csv"  # Replace with your file paths
    excel_file2_path = "Ashley_code/HEI.csv"
    compare_hei_columns(excel_file1_path, excel_file2_path)
