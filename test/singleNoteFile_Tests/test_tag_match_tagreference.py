import unittest
import os
import re

class TestTagMatchTagReference(unittest.TestCase):
    def setUp(self):
        # Calculate the path to the file relative to this test file.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        self.file_path = os.path.join(project_root, "singleNoteFIle.txt")
        self.assertTrue(os.path.exists(self.file_path), f"File {self.file_path} not found.")

    def test_tag_match_tagreference(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]

        # 1. Collect all tags from #### lines
        # Rule: Skip Example lines if they exist
        tags_defined = set()
        for i, line in enumerate(lines):
            if line.startswith("####") and not line.startswith("#####"):
                content = line[4:].strip()
                # Skip template/example lines
                if "Example:" in content or "tag1 tag2" in content:
                    continue
                # Tags are space-separated
                tags = content.split()
                for t in tags:
                    tags_defined.add(t)

        # 2. Collect all references from //// lines
        # Rule: Followed by non-empty, non-forwardslash, non-hashtag text
        references_with_content = set()
        for i, line in enumerate(lines):
            if line.startswith("////"):
                ref_tag = line[4:].strip()
                if not ref_tag:
                    continue
                
                # Check for following content
                has_content = False
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    if not next_line:
                        continue
                    if next_line.startswith("/") or next_line.startswith("#"):
                        break
                    else:
                        has_content = True
                        break
                
                if has_content:
                    references_with_content.add(ref_tag)

        # 3. Assert that each tag has a matching reference with content
        for tag in tags_defined:
            self.assertIn(tag, references_with_content, f"Tag '#### {tag}' does not have a matching '//// {tag}' with non-empty content following it.")

if __name__ == "__main__":
    unittest.main()
