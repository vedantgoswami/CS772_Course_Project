# Open the initial prompt file and read its content
with open("../initial_prompt.txt", "r") as file:
    content = file.read()

# Find the start and end positions of the code block
start_pos = content.find("```")
end_pos = content.find("```", start_pos + 3)

# Extract the code block content
code_block = content[start_pos + 3:end_pos]

# Write the extracted code block to a new file
file_path="../psuedoCode.txt"
with open(file_path, "w+") as file:
    file.write(code_block)
