import pandas as pd
import logging
from typing import Dict, List
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)


class DataLoader:
    """Enhanced data loader with multi-sheet support and document creation"""
    
    @staticmethod
    def load_excel(path) -> pd.DataFrame:
        """Load single sheet Excel file"""
        try:
            df = pd.read_excel(path)
            if df.empty:
                raise ValueError("Excel file is empty")
            logger.info(f"Loaded Excel with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading Excel: {e}")
            raise
    
    @staticmethod
    def load_all_sheets(path) -> Dict[str, pd.DataFrame]:
        """Load all sheets from Excel file"""
        try:
            sheets_dict = pd.read_excel(path, sheet_name=None)
            logger.info(f"Loaded {len(sheets_dict)} sheets from Excel")
            
            # Log sheet names and row counts
            for sheet_name, df in sheets_dict.items():
                logger.info(f"  - {sheet_name}: {len(df)} rows")
            
            return sheets_dict
        except Exception as e:
            logger.error(f"Error loading Excel sheets: {e}")
            # Fallback to single sheet
            return {"Sheet1": DataLoader.load_excel(path)}
    
    @staticmethod
    def create_documents_from_sheets(sheets: Dict[str, pd.DataFrame]) -> List[Document]:
        """Convert all sheets into LangChain documents for vector storage"""
        documents = []
        
        for sheet_name, df in sheets.items():
            for idx, row in df.iterrows():
                # Create rich text content from all columns
                content_parts = []
                metadata = {"sheet": sheet_name, "row_id": idx}
                
                for col in df.columns:
                    value = row[col]
                    if pd.notna(value):
                        content_parts.append(f"{col}: {value}")
                        metadata[col.lower()] = str(value)
                
                # Combine all information
                content = " | ".join(content_parts)
                
                documents.append(
                    Document(
                        page_content=content,
                        metadata=metadata
                    )
                )
        
        logger.info(f"Created {len(documents)} documents from all sheets")
        return documents
    
    @staticmethod
    def extract_entities(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Extract entities for graph building"""
        entities = {
            "diseases": [],
            "medicines": [],
            "symptoms": [],
            "procedures": [],
            "other": []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            values = df[col].dropna().unique().tolist()
            
            if "disease" in col_lower or "diagnosis" in col_lower:
                entities["diseases"].extend([str(v) for v in values])
            elif "medicine" in col_lower or "drug" in col_lower or "medication" in col_lower:
                entities["medicines"].extend([str(v) for v in values])
            elif "symptom" in col_lower or "sign" in col_lower:
                entities["symptoms"].extend([str(v) for v in values])
            elif "procedure" in col_lower or "treatment" in col_lower:
                entities["procedures"].extend([str(v) for v in values])
            else:
                entities["other"].extend([str(v) for v in values])
        
        return entities