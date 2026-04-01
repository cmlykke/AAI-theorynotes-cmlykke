import unittest
import os
import re

class TestTwoAndThreeHashUnique(unittest.TestCase):
    def setUp(self):
        # Calculate the path to the file relative to this test file.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        self.file_path = os.path.join(project_root, "singleNoteFIle.txt")
        self.assertTrue(os.path.exists(self.file_path), f"File {self.file_path} not found.")

    def test_two_and_three_hash_unique(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        seen_content = {}  # Normalized content -> (line_number, original_line)
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Rule: Once a line starting with //// is encountered, stop checking for duplicates
            if line_stripped.startswith("////"):
                break
            
            # Check if line starts with exactly ## or ###
            # Match ## or ### but not ####
            match = re.match(r"^(#{2,3})(?![#])", line_stripped)
            if match:
                # Content after hashes
                hash_part = match.group(1)
                content_part = line_stripped[len(hash_part):].strip()
                
                # Normalized content: remove all whitespace to handle trimmed/untrimmed/whitespace-in-middle
                normalized_content = "".join(content_part.split())
                
                if normalized_content in seen_content:
                    prev_line_num, prev_line_text = seen_content[normalized_content]
                    self.fail(f"Duplicate content found at line {i+1}: '{line.strip()}' matches line {prev_line_num}: '{prev_line_text.strip()}'. Normalized content: '{normalized_content}'")
                
                seen_content[normalized_content] = (i + 1, line)

if __name__ == "__main__":
    unittest.main()
