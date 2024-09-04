def has_lines_no_start_with_role(file_path):
    count_non_role_lines = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip().startswith('<ROLE>'):
                count_non_role_lines += 1
                print(f"Line {line_number} does not start with <ROLE>.")

    if count_non_role_lines > 0:
        return True
    return False
