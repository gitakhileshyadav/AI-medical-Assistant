import json
import re
from typing import List, Dict, Set, Any
from collections import OrderedDict

class CleanJSONTextExtractor:
    """
    Clean extractor that avoids duplicates and produces coherent natural language.
    """
    
    def __init__(self):
        self.extracted_data = {
            'patient_info': OrderedDict(),
            'clinical_info': [],
            'sites': OrderedDict(),
            'diagnoses': [],
            'descriptions': [],
            'other': []
        }
    
    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract and organize text from JSON-like string.
        
        Args:
            text: JSON string (complete or incomplete)
            
        Returns:
            Organized dictionary of extracted information
        """
        # Reset data
        self.extracted_data = {
            'patient_info': OrderedDict(),
            'clinical_info': [],
            'sites': OrderedDict(),
            'diagnoses': [],
            'descriptions': [],
            'other': []
        }
        
        # Clean and normalize the text
        text = self._clean_text(text)
        
        # Extract patient information
        self._extract_patient_info(text)
        
        # Extract clinical data
        self._extract_clinical_data(text)
        
        # Extract sites and specimens
        self._extract_sites_and_specimens(text)
        
        # Extract diagnoses
        self._extract_diagnoses(text)
        
        # Extract descriptions
        self._extract_descriptions(text)
        
        # Extract any other structured data
        self._extract_other_data(text)
        
        return self.extracted_data
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize the input text."""
        # Remove excessive whitespace but preserve structure
        text = re.sub(r'\s+', ' ', text)
        # Fix common broken patterns
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r'\[\s*,', '[', text)
        text = re.sub(r',\s*\]', ']', text)
        return text.strip()
    
    def _extract_patient_info(self, text: str):
        """Extract patient information."""
        patterns = {
            'patient_name': r'"patient_name"\s*:\s*"([^"]+)"',
            'patient_id': r'"patient_id"\s*:\s*"([^"]+)"',
            'age': r'"age"\s*:\s*"([^"]+)"',
            'sex': r'"sex"\s*:\s*"([^"]+)"'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                self.extracted_data['patient_info'][key] = match.group(1)
    
    def _extract_clinical_data(self, text: str):
        """Extract clinical data."""
        # Look for clinical_data array
        clinical_match = re.search(r'"clinical_data"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if clinical_match:
            clinical_text = clinical_match.group(1)
            # Extract individual items
            items = re.findall(r'"([^"]+)"', clinical_text)
            self.extracted_data['clinical_info'] = list(OrderedDict.fromkeys(items))
    
    def _extract_sites_and_specimens(self, text: str):
        """Extract site and specimen information."""
        # Look for specimen array
        specimen_match = re.search(r'"specimen"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if specimen_match:
            specimen_text = specimen_match.group(1)
            # Find all site-type pairs
            sites = re.findall(r'"site"\s*:\s*"([^"]+)"', specimen_text)
            types = re.findall(r'"type"\s*:\s*"([^"]+)"', specimen_text)
            
            # Pair them up
            for i in range(min(len(sites), len(types))):
                site_key = f"specimen_{i+1}"
                self.extracted_data['sites'][site_key] = {
                    'site': sites[i],
                    'type': types[i]
                }
    
    def _extract_diagnoses(self, text: str):
        """Extract diagnosis information."""
        # Look for diagnosis array
        diag_match = re.search(r'"diagnosis"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if diag_match:
            diag_text = diag_match.group(1)
            # Split into individual diagnosis objects
            diag_objects = re.findall(r'\{(.*?)\}', diag_text, re.DOTALL)
            
            for obj_text in diag_objects:
                site_match = re.search(r'"site"\s*:\s*"([^"]+)"', obj_text)
                type_match = re.search(r'"type"\s*:\s*"([^"]+)"', obj_text)
                result_match = re.search(r'"result"\s*:\s*"([^"]+)"', obj_text)
                
                if site_match and result_match:
                    diagnosis = {
                        'site': site_match.group(1),
                        'type': type_match.group(1) if type_match else 'N/A',
                        'result': result_match.group(1)
                    }
                    self.extracted_data['diagnoses'].append(diagnosis)
    
    def _extract_descriptions(self, text: str):
        """Extract description text."""
        # Look for gross_description array
        desc_match = re.search(r'"gross_description"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if desc_match:
            desc_text = desc_match.group(1)
            # Find description fields
            descriptions = re.findall(r'"description"\s*:\s*"([^"]+)"', desc_text)
            self.extracted_data['descriptions'] = list(OrderedDict.fromkeys(descriptions))
    
    def _extract_other_data(self, text: str):
        """Extract any other key-value pairs."""
        # Find all key-value pairs
        kv_pattern = r'"([^"]+)"\s*:\s*"([^"]+)"'
        matches = re.findall(kv_pattern, text)
        
        known_keys = ['patient_name', 'patient_id', 'age', 'sex', 'site', 'type', 'result', 'description']
        for key, value in matches:
            if key.lower() not in [k.lower() for k in known_keys]:
                if key not in self.extracted_data['patient_info']:
                    self.extracted_data['other'].append(f"{key}: {value}")
    
    def to_natural_language(self) -> str:
        """Convert extracted data to natural language."""
        lines = []
        
        # Patient Information
        if self.extracted_data['patient_info']:
            lines.append("PATIENT INFORMATION:")
            for key, value in self.extracted_data['patient_info'].items():
                display_key = key.replace('_', ' ').title()
                lines.append(f"  • {display_key}: {value}")
            lines.append("")
        
        # Clinical Information
        if self.extracted_data['clinical_info']:
            lines.append("CLINICAL DATA:")
            for item in self.extracted_data['clinical_info']:
                lines.append(f"  • {item}")
            lines.append("")
        
        # Sites and Specimens
        if self.extracted_data['sites']:
            lines.append("SPECIMEN COLLECTION SITES:")
            for key, data in self.extracted_data['sites'].items():
                lines.append(f"  • {data['site']} - {data['type']}")
            lines.append("")
        
        # Diagnoses
        if self.extracted_data['diagnoses']:
            lines.append("DIAGNOSIS RESULTS:")
            for i, diag in enumerate(self.extracted_data['diagnoses'], 1):
                lines.append(f"  {i}. Site: {diag['site']}")
                lines.append(f"     Type: {diag['type']}")
                lines.append(f"     Result: {diag['result']}")
                lines.append("")
        
        # Descriptions
        if self.extracted_data['descriptions']:
            lines.append("GROSS DESCRIPTIONS:")
            for i, desc in enumerate(self.extracted_data['descriptions'], 1):
                lines.append(f"  {i}. {desc}")
            lines.append("")
        
        # Other Information
        if self.extracted_data['other']:
            lines.append("ADDITIONAL INFORMATION:")
            for item in self.extracted_data['other']:
                lines.append(f"  • {item}")
        
        return "\n".join(lines)

# ============================================
# SIMPLE, CLEAN PIPELINE
# ============================================

def extract_text_from_json(json_input: Any) -> str:
    """
    Main function: Extract clean natural language from JSON.
    
    Args:
        json_input: JSON string, dict, list, or broken JSON
        
    Returns:
        Clean natural language text
    """
    # Convert to string if needed
    if isinstance(json_input, (dict, list)):
        try:
            json_str = json.dumps(json_input, indent=2)
        except:
            json_str = str(json_input)
    else:
        json_str = str(json_input)
    
    # Create extractor and process
    extractor = CleanJSONTextExtractor()
    extractor.extract_from_text(json_str)
    
    # Get natural language output
    natural_text = extractor.to_natural_language()
    
    return natural_text

# ============================================
# TEST WITH YOUR BROKEN JSON
# ============================================

def main():
    """Test the clean extractor with your broken JSON."""
    
    # Your incomplete JSON
    broken_json = '''{ "patient_name": "Yashvi M. Patel", "age": "21 Years", "sex": "Female", "patient_id": "556", "specimen": [ { "site": "Right Arm", "type": "Shave Biopsy" }, { "site": "Left Neck", "type": "Shave Biopsy" } ], "clinical_data": [ "R/O WART", "R/O TINEA" ], "diagnosis": [ { "site": "Skin, Right Arm", "type": "Shave Biopsy", "result": "Compatible with perforating disorder with features of elastosis perforans serpiginosa." }, { "site": "Skin, Left Neck", "type": "Shave Biopsy", "result": "Compatible with perforating disorder with features of elastosis perforans serpiginosa. Associated spongiotic dermatitis with occasional eosinophils." } ], "gross_description": [ { "site": "Right Arm", "description": "Received in formalin in a container labeled with the patient's name and 'R arm' is a single 0.5 x 0.4 x 0.1 cm irregular light grey-tan rough portion of tissue." }, { "site": "Left.,'''
    
    print("=" * 70)
    print("CLEAN NATURAL LANGUAGE EXTRACTION")
    print("=" * 70)
    
    # Store the result in a variable
    clean_natural_text = extract_text_from_json(broken_json)
    
    print("\nEXTRACTED NATURAL LANGUAGE TEXT:")
    print("=" * 70)
    
    # Print the clean text
    print(clean_natural_text)
    
    # Show variable info
    print("\n" + "=" * 70)
    print("VARIABLE INFORMATION:")
    print("=" * 70)
    print(f"Variable type: {type(clean_natural_text)}")
    print(f"Variable length: {len(clean_natural_text)} characters")
    print(f"Lines: {clean_natural_text.count(chr(10)) + 1}")
    
    return clean_natural_text

# ============================================
# QUICK USAGE FUNCTIONS
# ============================================

def json_to_text(json_input: Any) -> str:
    """One-line function for easy use."""
    return extract_text_from_json(json_input)

def process_json_file(filename: str) -> str:
    """Process a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        return extract_text_from_json(content)
    except Exception as e:
        return f"Error reading file: {str(e)}"

# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    # Run the main test
    extracted_text = main()
    
    # Demonstrate usage
    print("\n" + "=" * 70)
    print("HOW TO USE:")
    print("=" * 70)
    
    # Example 1: Direct usage
    print("\nExample 1 - Direct usage:")
    print("-" * 40)
    sample_json = '{"patient_name": "John Doe", "age": "30 Years"}'
    result = json_to_text(sample_json)
    print(result)
    
    # Example 2: Process file
    print("\nExample 2 - File processing:")
    print("-" * 40)
    print("""
# Save your JSON to a file first
with open('data.json', 'w') as f:
    f.write(your_json_string)

# Then process it
text = process_json_file('data.json')
print(text)
""")
    
    # Example 3: Using the variable
    print("\nExample 3 - Using the extracted text variable:")
    print("-" * 40)
    print("""
# The text is stored in 'extracted_text' variable
print(extracted_text)  # Prints the clean text

# Save to file
with open('output.txt', 'w') as f:
    f.write(extracted_text)

# Use in other functions
def analyze_text(text):
    words = text.split()
    print(f"Word count: {len(words)}")

analyze_text(extracted_text)
""")
    
    # Show actual variable usage
    print("\nActual variable usage demo:")
    print(f"First 100 chars: {extracted_text[:100]}...")