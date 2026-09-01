import json
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# Configure basic logging to provide clear feedback
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def parse_txt(file_path: Path) -> List[Dict[str, str]]:
    """
    Reads a text file and extracts Question & Answer pairs based on heuristics.
    """
    items: List[Dict[str, str]] = []
    q: Optional[str] = None
    
    try:
        # Iterating line-by-line prevents memory crashes on large files
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Heuristics: Treat lines ending with '?', ':', or starting with '- ' as questions
                if line.endswith('?') or line.endswith(':') or line.startswith('- '):
                    q = line.lstrip('- ').rstrip(':').strip()
                    continue
                
                # Otherwise, treat the line as an answer
                if q is None:
                    # If file is just answers, use an auto-incrementing index as a question placeholder
                    items.append({"question": f"Q{len(items) + 1}", "answer": line})
                else:
                    ans = line
                    if ans.lower().startswith('answer:'):
                        # Safely split only on the first colon
                        ans = ans.split(':', 1)[1].strip()
                        
                    items.append({"question": q, "answer": ans})
                    q = None  # Reset for the next question

    except PermissionError:
        logging.error(f"Permission denied: Cannot read the file '{file_path}'.")
        sys.exit(1)
    except OSError as e:
        logging.error(f"An OS error occurred while reading '{file_path}': {e}")
        sys.exit(1)
        
    return items

def txt_to_json(txt_path: Path, json_path: Path) -> None:
    """
    Parses a text file and saves the parsed Q&A pairs as a JSON file.
    """
    # 1. Validate Input File
    if not txt_path.is_file():
        logging.error(f"Input file does not exist or is a directory: '{txt_path}'")
        sys.exit(1)

    # 2. Parse Text
    items = parse_txt(txt_path)
    if not items:
        logging.warning("No items were parsed from the input file. Output will be empty.")

    # 3. Format Output
    out =[
        {"id": i, "question": it.get("question", ""), "answer": it.get("answer", "")}
        for i, it in enumerate(items, start=1)
    ]

    # 4. Safely Write to JSON
    try:
        # Secure: Ensure parent directories exist before writing
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with json_path.open('w', encoding='utf-8') as f:
            # ensure_ascii=False ensures special characters (like emojis) aren't escaped
            json.dump(out, f, indent=2, ensure_ascii=False)
            
        logging.info(f"Successfully wrote {len(out)} items to '{json_path}'")
        
    except PermissionError:
        logging.error(f"Permission denied: Cannot write to '{json_path}'.")
        sys.exit(1)
    except OSError as e:
        logging.error(f"An OS error occurred while writing '{json_path}': {e}")
        sys.exit(1)

def main():
    # argparse automatically handles incorrect argv counts, types, and provides --help
    parser = argparse.ArgumentParser(
        description="Convert a Text file containing Q&A into a structured JSON format."
    )
    parser.add_argument(
        "input_file", 
        type=Path, 
        help="Path to the input text file (e.g., input.txt)"
    )
    parser.add_argument(
        "output_file", 
        type=Path, 
        help="Path to the output JSON file (e.g., output.json)"
    )
    
    args = parser.parse_args()

    # Resolve paths to handle absolute/relative directory traversal securely
    txt_to_json(args.input_file.resolve(), args.output_file.resolve())

if __name__ == "__main__":
    main()