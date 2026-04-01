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

    def _get_content_blocks(self, lines):
        """Helper to parse content blocks for ## and ### tags."""
        blocks = []
        i = 0
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()
            
            # Rule: Once a line starting with //// is encountered, stop
            if line_stripped.startswith("////"):
                break
            
            # Check if line starts with exactly ## or ###
            match = re.match(r"^(#{2,3})(?![#])", line_stripped)
            if match:
                hash_part = match.group(1)
                content_lines = [line_stripped[len(hash_part):].strip()]
                start_line_num = i + 1
                start_line_text = line
                
                # Accumulate subsequent lines
                j = i + 1
                while j < len(lines):
                    next_line_stripped = lines[j].strip()
                    if not next_line_stripped:
                        j += 1
                        continue
                    if next_line_stripped.startswith("#") or next_line_stripped.startswith("////"):
                        break
                    content_lines.append(next_line_stripped)
                    j += 1
                
                full_content = " ".join(content_lines)
                normalized_content = "".join(full_content.split())
                
                if normalized_content:
                    blocks.append({
                        "hash": hash_part,
                        "normalized": normalized_content,
                        "line_num": start_line_num,
                        "line_text": start_line_text
                    })
                i = j
                continue
            i += 1
        return blocks

    def test_two_hash_unique(self):
        """Verify that ## headers (with their content) are unique."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        blocks = self._get_content_blocks(lines)
        seen_two = {}
        for b in blocks:
            if b["hash"] == "##":
                norm = b["normalized"]
                if norm in seen_two:
                    prev = seen_two[norm]
                    self.fail(f"Duplicate ## content found starting at line {b['line_num']}: '{b['line_text'].strip()}' matches line {prev['line_num']}: '{prev['line_text'].strip()}'.")
                seen_two[norm] = b

    def test_three_hash_unique(self):
        """Verify that ### headers (with their content) are unique."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        blocks = self._get_content_blocks(lines)
        seen_three = {}
        for b in blocks:
            if b["hash"] == "###":
                norm = b["normalized"]
                if norm in seen_three:
                    prev = seen_three[norm]
                    self.fail(f"Duplicate ### content found starting at line {b['line_num']}: '{b['line_text'].strip()}' matches line {prev['line_num']}: '{prev['line_text'].strip()}'.")
                seen_three[norm] = b

    def test_two_and_three_hash_cross_unique(self):
        """Verify that content is unique across both ## and ### headers."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        blocks = self._get_content_blocks(lines)
        seen_all = {}
        for b in blocks:
            norm = b["normalized"]
            if norm in seen_all:
                prev = seen_all[norm]
                self.fail(f"Duplicate content found across headers at line {b['line_num']} ('{b['hash']}') matches line {prev['line_num']} ('{prev['hash']}').")
            seen_all[norm] = b

if __name__ == "__main__":
    unittest.main()
