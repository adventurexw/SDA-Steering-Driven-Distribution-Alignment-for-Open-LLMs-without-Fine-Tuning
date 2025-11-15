import json
import os
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm

class JsonlHandler:
    """Utility class for handling JSONL files in JSONL format."""
    
    def __init__(self, file_path: str, auto_create_dir: bool = True):
        """
        Initialize the JSONL file handler.
        
        Args:
            file_path: Path to the JSONL file.
            auto_create_dir: Whether to automatically create missing directories.
        """
        self.file_path = file_path
        
        if auto_create_dir:
            dir_path = os.path.dirname(file_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
    
    def append(self, data: Dict[str, Any]) -> None:
        """
        Append a record to the JSONL file.
        
        Args:
            data: The data to append; should be a dictionary.
        """
        # Ensure data contains required fields (optional)
        required_fields = [
            "query", "original_response", "O_score",
            "prompt_response", "P_score",
            "correction_sdaJS", "C_score",
            "correction_aligner", "A_score"
        ]
        
        for field in required_fields:
            if field not in data:
                data[field] = ""  # Fill missing fields with empty values
        
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    def read_all(self) -> List[Dict[str, Any]]:
        """
        Read all records from the JSONL file.
        
        Returns:
            A list containing all records.
        """
        if not os.path.exists(self.file_path):
            return []
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    
    def get_line_count(self) -> int:
        """Get the number of lines (i.e., number of records) in the file."""
        if not os.path.exists(self.file_path):
            return 0
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    
    def update_record(self, id_value: Any, new_data: Dict[str, Any]) -> bool:
        """
        Update a record by its ID.
        
        Args:
            id_value: The ID value of the record.
            new_data: The new data used to update the record.
            
        Returns:
            Whether the update succeeded.
        """
        if not os.path.exists(self.file_path):
            return False
        
        updated = False
        temp_path = self.file_path + '.temp'
        
        with open(self.file_path, 'r', encoding='utf-8') as f, \
             open(temp_path, 'w', encoding='utf-8') as out_f:
            
            for line in f:
                record = json.loads(line)
                if record.get('id') == id_value:
                    record.update(new_data)
                    updated = True
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        if updated:
            os.replace(temp_path, self.file_path)
        else:
            os.remove(temp_path)
            
        return updated
    
    def analyze_data(self) -> Dict[str, Any]:
        """Analyze dataset statistics."""
        records = self.read_all()
        if not records:
            return {"message": "No data to analyze"}
        
        # Compute statistics such as average lengths and scores
        stats = {
            "total_records": len(records),
            "avg_query_length": sum(len(r.get("query", "")) for r in records) / len(records),
            "avg_original_response_length": sum(len(r.get("original_response", "")) for r in records) / len(records),
            "avg_prompt_response_length": sum(len(r.get("prompt_response", "")) for r in records) / len(records),
            "avg_correction_sdaJS_length": sum(len(r.get("correction_sdaJS", "")) for r in records) / len(records),
            "avg_correction_aligner_length": sum(len(r.get("correction_aligner", "")) for r in records) / len(records),
            "avg_scores": {
                "O_score": sum(r.get("O_score", 0) for r in records) / len(records),
                "P_score": sum(r.get("P_score", 0) for r in records) / len(records),
                "C_score": sum(r.get("C_score", 0) for r in records) / len(records),
                "A_score": sum(r.get("A_score", 0) for r in records) / len(records)
            }
        }
        
        return stats
    
    def deduplicate(self) -> int:
        """
        Deduplicate records (based on ID).
        
        Returns:
            The number of duplicate records removed.
        """
        if not os.path.exists(self.file_path):
            return 0
        
        unique_records = {}
        removed = 0
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                id_val = record.get('id')
                if id_val in unique_records:
                    removed += 1
                else:
                    unique_records[id_val] = record
        
        if removed > 0:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                for record in unique_records.values():
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        return removed

# Example usage in main scripts
def save_results_to_jsonl(
    query: str, original_response: str, O_score: Union[int, float], 
    prompt_response: str, P_score: Union[int, float],
    correction_sdaJS: str, C_score: Union[int, float],
    correction_aligner: str, A_score: Union[int, float],
    file_path: str, id_value: Optional[int] = None
):
    """
    Save results to a JSONL file.
    
    Args:
        query: The query text.
        original_response: The original response.
        O_score: Score of the original response.
        prompt_response: The prompt-based response.
        P_score: Score of the prompt-based response.
        correction_sdaJS: The corrected response by sdaJS method.
        C_score: Score of the sdaJS correction.
        correction_aligner: The corrected response by aligner method.
        A_score: Score of the aligner correction.
        file_path: Path to the JSONL file.
        id_value: Optional ID value; if None, use (line count + 1) as ID.
    """
    handler = JsonlHandler(file_path)
    
    # If ID is not provided, use current line count + 1 as the ID
    if id_value is None:
        id_value = handler.get_line_count() + 1
    
    data = {
        "id": id_value,
        "query": query,
        "original_response": original_response,
        "O_score": O_score,
        "prompt_response": prompt_response,
        "P_score": P_score,
        "correction_sdaJS": correction_sdaJS,
        "C_score": C_score,
        "correction_aligner": correction_aligner,
        "A_score": A_score
    }
    
    handler.append(data)


# Utility: read results from a JSONL file and pretty print the first N
def print_results_from_jsonl(file_path: str, limit: int = 5):
    """Read results from a JSONL file and print the first N records."""
    handler = JsonlHandler(file_path)
    records = handler.read_all()
    
    print(f"Read {len(records)} records from {file_path}")
    
    for i, record in enumerate(records[:limit]):
        print(f"\n=== Record {i+1} ===")
        print(f"ID: {record['id']}")
        print(f"Query: {record['query']}")
        print(f"Original response: {record['original_response']}")
        print(f"Original score: {record['O_score']}")
        print(f"Prompt response: {record['prompt_response']}")
        print(f"Prompt score: {record['P_score']}")
        print(f"sdaJS correction: {record['correction_sdaJS']}")
        print(f"sdaJS score: {record['C_score']}")
        print(f"aligner correction: {record['correction_aligner']}")
        print(f"aligner score: {record['A_score']}")

# Utility: analyze statistics in a JSONL results file
def analyze_results(file_path: str):
    """Analyze summary statistics of a JSONL results file."""
    handler = JsonlHandler(file_path)
    stats = handler.analyze_data()
    
    print("\n=== Summary statistics ===")
    for key, value in stats.items():
        if key == "avg_scores":
            print("Average scores:")
            for score_key, score_value in value.items():
                print(f"  - {score_key}: {score_value:.2f}")
        else:
            print(f"{key}: {value}")