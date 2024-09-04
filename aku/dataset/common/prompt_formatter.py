def format_prompt(contents):
    """Format the contents into a prompt.

    Args:
        contents (list): A list of dictionaries containing the role and content of the conversation.

    Returns:
        str: The formatted prompt.
    """
    
    prompt = ""
    for i, content in enumerate(contents):
        if i == 0:
            prompt += f"<ROLE>{content['role']}</ROLE>__BR__{content['content']}"
        else:
            prompt += f"__BR__<ROLE>{content['role']}</ROLE>__BR__{content['content']}"

    prompt += "<EOS>\n"

    return prompt
