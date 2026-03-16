import re
import os

def extract_row_col(input_string):
    """
    Extracts the row and column integers from a string in the format 
    'UOP-001_roi_1000_row_21600_col_28080'.
    
    Args:
        input_string (str): The input string containing row and column information.
        
    Returns:
        tuple: A tuple containing (row, col) as integers.
    """
    # Use regular expressions to find row and column
    row_match = re.search(r"row_(\d+)", input_string)
    col_match = re.search(r"col_(\d+)", input_string)
    
    if row_match and col_match:
        row = int(row_match.group(1))
        col = int(col_match.group(1))
        return row, col
    else:
        raise ValueError("The input string does not contain valid row or column information.")

def read_sample_files(sample_path):
    with open(os.path.join(sample_path), 'r') as file:
        lines = file.readlines()
    
    out_name_list = []
    lines = [line.strip() for line in lines]
    for sample_line in lines:
        out_name_list.append(sample_line)
    
    return out_name_list
        