import re
import pandas as pd
from pathlib import Path
from docx import Document

def read_docx_text(path):
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return paragraphs

def parse_transcript(paragraphs):
    agent_pattern = re.compile(r"^[a-z_]+_(agent|proxy)$", re.IGNORECASE)
    token_pattern = re.compile(r"^Tokens:\s*\d{3,5}\s*$")
    
    records = []
    current_actor = None
    current_message_lines = []
    
    def flush():
        if current_actor is not None and current_message_lines:
            filtered_lines = [
                line for line in current_message_lines 
                if not token_pattern.match(line.strip())
            ]
            msg = "\n".join(filtered_lines).strip()
            if msg:
                records.append({"message": msg, "actor": current_actor})
    
    for para in paragraphs:
        line = para.strip()
        
        if agent_pattern.match(line):
            flush()
            current_message_lines = []
            current_actor = line
        else:
            if current_actor is not None:
                current_message_lines.append(para)
    
    flush()
    return records

def to_excel(records, out_path):
    df = pd.DataFrame(records, columns=["message", "actor"])
    
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Conversation", index=False)
        
        # Format for readability
        worksheet = writer.sheets["Conversation"]
        worksheet.column_dimensions["A"].width = 100
        worksheet.column_dimensions["B"].width = 25
    
    return len(records)

def process_file(docx_path):
    print(f"Processing: {docx_path.name}")
    
    try:
        paragraphs = read_docx_text(docx_path)
        records = parse_transcript(paragraphs)
        
        output_path = docx_path.parent / f"{docx_path.stem}.xlsx"
        
        num_turns = to_excel(records, output_path)
            
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def main():
    current_dir = Path(__file__).parent
    docx_files = list(current_dir.glob("*.docx"))
    
    docx_files = [f for f in docx_files if not f.name.startswith("~$")]
    
    if not docx_files:
        print("No .docx files found in current directory.")
        return
    
    print(f"Found {len(docx_files)} .docx file(s):")
    for f in docx_files:
        print(f"  • {f.name}")
    
    success_count = 0
    for docx_file in docx_files:
        if process_file(docx_file):
            success_count += 1
    
    print(f"{success_count}/{len(docx_files)} files converted successfully")

if __name__ == "__main__":
    main()
