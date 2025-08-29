def remove_numbers_and_linebreaks(file_path):
    """Removes numbers and line breaks from a UTF-8 text file.

    Args:
        file_path: The path to the input text file.
    """

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Remove numbers and line breaks
    filtered_text = ''.join(char for char in text if not char.isdigit() and char != '\n')

    with open('output.txt', 'w', encoding='utf-8') as out_file:
        out_file.write(filtered_text)


# Example usage
file_path = "input.txt"
remove_numbers_and_linebreaks(file_path)


