import unittest
import os
import re

class TestHashtagOrderInBlocks(unittest.TestCase):
    def setUp(self):
        # Set the path to the file to be tested.
        # Assuming the test is run from the project root.
        self.file_path = "singleNoteFile.txt"
        self.assertTrue(os.path.exists(self.file_path), f"File {self.file_path} not found.")

    def test_file_content_constraints(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        current_subsection_hashes = []
        last_hash_len = 0
        
        # Helper function to validate and record a subsection's requirements
        def validate_subsection(hashes):
            if not hashes:
                return
            # Each subsection must have a # string of length 2 and a # string of length 3
            self.assertIn(2, hashes, f"Subsection {hashes} missing ##")
            self.assertIn(3, hashes, f"Subsection {hashes} missing ###")

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                # Blank lines are allowed and don't affect hash sequence
                continue
            
            # Each line should start with 1 to 6 # symbols
            match = re.match(r"^(#+)", line_stripped)
            self.assertIsNotNone(match, f"Line {i+1} does not start with # symbols: '{line_stripped}'")
            
            hash_str = match.group(1)
            hash_len = len(hash_str)
            self.assertTrue(1 <= hash_len <= 6, f"Line {i+1} hash length {hash_len} is not between 1 and 6.")

            # Rule: Each subsequent string of # symbols should be longer than the previous one.
            # If a string is shorter than a previous one, the length must either be 1 (#) or 2 (##) marking the start of a new subsection.
            
            if last_hash_len > 0:
                if hash_len <= last_hash_len:
                    # Starting a new subsection
                    # Allow starting a new subsection with 1 (#) or 2 (##)
                    self.assertIn(hash_len, [1, 2], f"Line {i+1}: Shorter or equal string length {hash_len} must be 1 (#) or 2 (##) to start a new subsection (previous length was {last_hash_len}).")
                    
                    # Validate the subsection that just ended
                    validate_subsection(current_subsection_hashes)
                    
                    # Reset for new subsection
                    current_subsection_hashes = []
                
            current_subsection_hashes.append(hash_len)
            last_hash_len = hash_len

        # Validate the last subsection
        validate_subsection(current_subsection_hashes)

if __name__ == "__main__":
    unittest.main()
