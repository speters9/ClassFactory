# general libraries used
import os
from pathlib import Path

from dotenv import load_dotenv
from utils.readings_to_txt import get_chapter_text

load_dotenv()

wd = Path(os.getenv('projectDir'))
COMPUTER = os.getenv('USER')

readingDir = Path(os.getenv('readingsDir'))

# %%
text_args = {
    'chapter_title': "Chapter 6: The Presidency",
    'section_titles': [
        "Chapter 6: The Presidency",
        "The Constitutional Origins and Powers of the Presidency",
        "The Rise of Presidential Government",
        "Presidential Government",
        "Conclusion"],
    'chapter_number': 6,
    'verbose': False,
}

text = get_chapter_text(**text_args)

# Save the concatenated text to a .txt file
output_file = Path(readingDir / "L13/13.1 textbook_chapter6.txt")
output_file.write_text(text)
