class AlphabetIterator:
    """Iterator for alphabet letters from 'A' to 'Z'.
    """
    def __init__(self):
        """Initialize the iterator.
        """
        self.current = ord('A') # ASCII code of 'A'
    
    def next(self) -> str:
        """Get the next letter in the alphabet.

        Returns:
            str: The next letter in the alphabet.
        """
        letter = chr(self.current)
        self.current += 1
        
        if self.current > ord('Z'):
            self.current = ord('A')
        
        return letter
