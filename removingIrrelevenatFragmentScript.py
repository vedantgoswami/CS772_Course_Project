import os
import re

############################################# Count the No of Functions in The Code ######################
    
def count_functions(code):
    # Define a regular expression pattern to match C/C++ function declarations
    pattern = re.compile(r'\b\w+\s+\w+\s*\(.*\)\s*{')
    
    # Find all matches in the code
    matches = pattern.findall(code)
    #print(matches)
    return matches
    
def filter_valid_data_types(lines):
    valid_data_types = ['void', 'int', 'char', 'float', 'double', 'long', 'short', 'unsigned', 'signed']

    filtered_lines = []
    for line in lines:
        # Define a regular expression pattern to match valid C/C++ data type declarations
        pattern = re.compile(r'\b(' + '|'.join(valid_data_types) + r')\b')
        
        # Check if the line starts with a valid data type
        if pattern.match(line):
            filtered_lines.append(line)

    return filtered_lines

######################################### Removing INT_MAIN #############################################
def remove_int_main(code):
    code+="$"
    # Define a regular expression pattern for int main() block
    pattern = re.compile(r'\s*int\s+main\s*\(\s*\)\s*{(.|\n)*?$')

    # Find and remove int main() block
    modified_code = re.sub(pattern, '', code)

    return modified_code

######################################## Removing Include and Namespace ##################################
def remove_include_namespace(code):
    # Define a regular expression pattern for lines with #include and using namespace
    pattern = re.compile(r'#include.*|using\s+namespace\s+\w+.*')

    # Remove lines with #include and using namespace
    modified_code = re.sub(pattern, '', code)

    return modified_code


############################################ Remove CIN/COUT ############################################
def remove_cin_cout_lines(code):
    # Define a regular expression pattern for lines with cin or cout
    pattern = re.compile(r'\b(cin|cout).*')

    # Remove lines with cin or cout
    modified_code = re.sub(pattern, '', code)

    return modified_code


###############################################  READ THE CPP/C Files ##############################################

count = 0
countl = 0

def read_cpp_c_files(directory):
    global count
    global countl
    for filename in os.listdir(directory):
        outputfilepath = os.path.join(directory,"modified.txt")
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and (filename.endswith(".cpp") or filename.endswith(".c")):
            with open(filepath, 'r') as file:
                code=file.read()
                matches=count_functions(code)
                valid_matches=filter_valid_data_types(matches)
                # print(code)
                # print(matches)
                if len(valid_matches) > 2 :
                    print(outputfilepath)
                    count+=1
                    m_code = remove_int_main(code)
                    modified_code = remove_include_namespace(m_code)
                    with open(outputfilepath, 'w') as out:
                        out.write(modified_code)
                else :
                    countl+=1
                    print(countl)
                    print("Within Main "+outputfilepath)
                    m_code = remove_cin_cout_lines(code)
                    modified_code = remove_include_namespace(m_code)
                    with open(outputfilepath, 'w') as out:
                        out.write(modified_code)
                     


# List directories in the current folder
for directory in os.listdir('./'):
    if os.path.isdir(directory):
        read_cpp_c_files(directory)
        # break  # Stop after the first directory
print(count)
print(countl)