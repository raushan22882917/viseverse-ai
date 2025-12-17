"""
Document Processing Service implementation using PaddleOCR.
Handles multilingual OCR processing, document type classification, and structured data extraction.
"""

import asyncio
import io
import logging
from typing import List, Optional, Dict, Any, Tuple
from uuid import uuid4
import json
import re
from datetime import datetime

import paddleocr
from PIL import Image
import numpy as np

from ...core.interfaces import DocumentProcessingService
from ...core.models import (
    ProcessedDocument,
    StructuredData,
    QualityScore,
    DocumentType,
    ProcessingStatus,
    LayoutData,
    TableData,
    DateField,
    FinancialData
)


logger = logging.getLogger(__name__)


class PaddleOCRDocumentProcessor(DocumentProcessingService):
    """
    Document processing service using PaddleOCR for multilingual text extraction.
    """
    
    def __init__(self, use_gpu: bool = False, lang: str = 'en'):
        """
        Initialize the document processor.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            lang: Default language for OCR processing
        """
        self.use_gpu = use_gpu
        self.default_lang = lang
        self._ocr_engine = None
        self._supported_languages = {
            'en', 'ch', 'ta', 'te', 'ka', 'ja', 'ko', 'hi', 'ar', 'fr', 'de', 'es', 'pt', 'ru', 'it'
        }
        
        # Document type classification patterns
        self._document_patterns = {
            DocumentType.PASSPORT: [
                r'passport', r'passeport', r'pasaporte', r'reisepass', r'passaporto',
                r'nationality', r'date of birth', r'place of birth', r'passport no'
            ],
            DocumentType.VISA_APPLICATION: [
                r'visa application', r'application form', r'immigration', r'entry permit'
            ],
            DocumentType.EMPLOYMENT_LETTER: [
                r'employment', r'job offer', r'work authorization', r'salary', r'position'
            ],
            DocumentType.BANK_STATEMENT: [
                r'bank statement', r'account balance', r'transaction', r'deposit', r'withdrawal'
            ],
            DocumentType.EDUCATIONAL_CERTIFICATE: [
                r'certificate', r'diploma', r'degree', r'university', r'graduation', r'academic'
            ],
            DocumentType.MEDICAL_CERTIFICATE: [
                r'medical', r'health', r'doctor', r'physician', r'examination', r'certificate'
            ]
        }
    
    def _get_ocr_engine(self, lang: str = None) -> paddleocr.PaddleOCR:
        """Get or create OCR engine instance."""
        if self._ocr_engine is None:
            use_lang = lang if lang in self._supported_languages else self.default_lang
            try:
                # Try with newer parameter names first
                self._ocr_engine = paddleocr.PaddleOCR(
                    use_textline_orientation=True,
                    lang=use_lang
                )
            except Exception:
                try:
                    # Fallback to older parameter names
                    self._ocr_engine = paddleocr.PaddleOCR(
                        use_angle_cls=True,
                        lang=use_lang
                    )
                except Exception:
                    # Minimal configuration as last resort
                    self._ocr_engine = paddleocr.PaddleOCR(lang=use_lang)
        return self._ocr_engine
    
    async def process_documents(
        self, 
        files: List[bytes], 
        filenames: List[str],
        language: Optional[str] = None
    ) -> List[ProcessedDocument]:
        """
        Process multiple documents using OCR.
        
        Args:
            files: List of document file contents as bytes
            filenames: List of original filenames
            language: Optional language hint for OCR processing
            
        Returns:
            List of processed documents with extracted data
        """
        if len(files) != len(filenames):
            raise ValueError("Number of files must match number of filenames")
        
        processed_docs = []
        
        for file_bytes, filename in zip(files, filenames):
            try:
                doc = await self._process_single_document(file_bytes, filename, language)
                processed_docs.append(doc)
            except Exception as e:
                logger.error(f"Failed to process document {filename}: {str(e)}")
                # Create failed document entry
                failed_doc = ProcessedDocument(
                    document_type=DocumentType.PASSPORT,  # Default fallback
                    extracted_text="",
                    layout_info=LayoutData(),
                    confidence_score=0.0,
                    language=language or self.default_lang,
                    processing_status=ProcessingStatus.FAILED
                )
                processed_docs.append(failed_doc)
        
        return processed_docs
    
    async def _process_single_document(
        self, 
        file_bytes: bytes, 
        filename: str, 
        language: Optional[str] = None
    ) -> ProcessedDocument:
        """Process a single document."""
        try:
            # Convert bytes to image
            image = Image.open(io.BytesIO(file_bytes))
            image_array = np.array(image)
            
            # Run OCR in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            ocr_engine = self._get_ocr_engine(language)
            
            ocr_result = await loop.run_in_executor(
                None, 
                ocr_engine.ocr, 
                image_array
            )
            
            # Extract text and layout information
            extracted_text, layout_info, confidence_score = self._extract_text_and_layout(ocr_result)
            
            # Classify document type
            document_type = self._classify_document_type(extracted_text, filename)
            
            # Extract table data if present
            table_data = self._extract_table_data(ocr_result)
            
            # Detect language
            detected_lang = self._detect_language(extracted_text, language)
            
            return ProcessedDocument(
                document_type=document_type,
                extracted_text=extracted_text,
                layout_info=layout_info,
                table_data=table_data,
                confidence_score=confidence_score,
                language=detected_lang,
                processing_status=ProcessingStatus.COMPLETED
            )
        except Exception as e:
            logger.error(f"Error in _process_single_document: {str(e)}")
            # Return a failed document with minimal info
            return ProcessedDocument(
                document_type=DocumentType.PASSPORT,  # Default fallback
                extracted_text="",
                layout_info=LayoutData(),
                confidence_score=0.0,
                language=language or self.default_lang,
                processing_status=ProcessingStatus.FAILED
            )
    
    def _extract_text_and_layout(self, ocr_result: List) -> Tuple[str, LayoutData, float]:
        """Extract text and layout information from OCR result."""
        if not ocr_result or not ocr_result[0]:
            return "", LayoutData(), 0.0
        
        # Debug: Log the OCR result structure
        logger.debug(f"OCR result type: {type(ocr_result)}")
        logger.debug(f"OCR result length: {len(ocr_result)}")
        if ocr_result and len(ocr_result) > 0:
            logger.debug(f"First element type: {type(ocr_result[0])}")
            if isinstance(ocr_result[0], list) and len(ocr_result[0]) > 0:
                logger.debug(f"First line: {ocr_result[0][0]}")
        
        text_parts = []
        bounding_boxes = []
        confidences = []
        
        try:
            # Handle different OCR result formats
            if isinstance(ocr_result[0], list) and len(ocr_result[0]) > 0:
                # Standard format: [[[bbox], [text, confidence]], ...]
                for line in ocr_result[0]:
                    if not line or len(line) < 2:
                        continue
                        
                    bbox = line[0]
                    text_info = line[1]
                    
                    # Handle different text_info formats
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text = str(text_info[0]) if text_info[0] is not None else ""
                        confidence = float(text_info[1]) if text_info[1] is not None else 0.5
                    elif isinstance(text_info, str):
                        text = text_info
                        confidence = 0.5  # Default confidence
                    else:
                        text = str(text_info) if text_info is not None else ""
                        confidence = 0.5
                    
                    # Skip empty text
                    if not text.strip():
                        continue
                    
                    text_parts.append(text)
                    confidences.append(confidence)
                    
                    # Store bounding box information
                    bounding_boxes.append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': confidence
                    })
            elif isinstance(ocr_result[0], dict):
                # Dictionary format - check for text fields
                if 'rec_texts' in ocr_result[0] and ocr_result[0]['rec_texts']:
                    for i, text in enumerate(ocr_result[0]['rec_texts']):
                        if text and text.strip():
                            confidence = ocr_result[0].get('rec_scores', [0.5])[i] if i < len(ocr_result[0].get('rec_scores', [])) else 0.5
                            text_parts.append(text)
                            confidences.append(confidence)
                            bounding_boxes.append({
                                'text': text,
                                'bbox': [],
                                'confidence': confidence
                            })
                # If no text found in dictionary, it's likely empty
            else:
                # Check if it's a string representation of debug info
                text_content = str(ocr_result[0]) if ocr_result[0] else ""
                # If it looks like debug info (contains 'input_path', 'model_settings', etc.), treat as empty
                if any(debug_key in text_content for debug_key in ['input_path', 'model_settings', 'rec_texts', 'dt_polys']):
                    # This is debug/metadata, not actual text
                    pass
                elif text_content.strip():
                    text_parts.append(text_content)
                    confidences.append(0.5)
                    bounding_boxes.append({
                        'text': text_content,
                        'bbox': [],
                        'confidence': 0.5
                    })
        except Exception as e:
            logger.error(f"Error extracting text from OCR result: {str(e)}")
            return "", LayoutData(), 0.0
        
        extracted_text = '\n'.join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        layout_info = LayoutData(
            bounding_boxes=bounding_boxes,
            text_regions=[{'text': text, 'confidence': conf} for text, conf in zip(text_parts, confidences)]
        )
        
        return extracted_text, layout_info, avg_confidence
    
    def _classify_document_type(self, text: str, filename: str) -> DocumentType:
        """Classify document type based on content and filename."""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Check filename first
        for doc_type, patterns in self._document_patterns.items():
            for pattern in patterns:
                if pattern in filename_lower:
                    return doc_type
        
        # Check content
        type_scores = {}
        for doc_type, patterns in self._document_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            type_scores[doc_type] = score
        
        # Return type with highest score, default to passport
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 0:
                return best_type
        
        return DocumentType.PASSPORT  # Default fallback
    
    def _extract_table_data(self, ocr_result: List) -> List[TableData]:
        """Extract table data from OCR result."""
        # Simplified table extraction - can be enhanced with more sophisticated logic
        tables = []
        
        try:
            if not ocr_result or not ocr_result[0]:
                return tables
            
            # Group text by vertical position to identify potential table rows
            lines_by_y = {}
            for line in ocr_result[0]:
                if len(line) >= 2:
                    bbox = line[0]
                    text_info = line[1]
                    
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text = text_info[0]
                        confidence = text_info[1]
                    else:
                        text = str(text_info)
                        confidence = 0.5
                    
                    # Use average Y coordinate as row identifier
                    y_pos = int((bbox[0][1] + bbox[2][1]) / 2)
                    
                    if y_pos not in lines_by_y:
                        lines_by_y[y_pos] = []
                    
                    lines_by_y[y_pos].append({
                        'text': text,
                        'x_pos': bbox[0][0],
                        'confidence': confidence
                    })
            
            # Simple table detection: if we have multiple rows with similar column structure
            if len(lines_by_y) >= 3:  # At least 3 rows for a table
                # Sort rows by Y position
                sorted_rows = sorted(lines_by_y.items())
                
                # Check if rows have similar number of columns
                row_lengths = [len(row[1]) for row in sorted_rows]
                if len(set(row_lengths)) <= 2:  # Allow some variation
                    # Extract table data
                    table_rows = []
                    for y_pos, row_data in sorted_rows:
                        # Sort columns by X position
                        sorted_cols = sorted(row_data, key=lambda x: x['x_pos'])
                        row_text = [col['text'] for col in sorted_cols]
                        table_rows.append(row_text)
                    
                    if table_rows:
                        # First row as headers, rest as data
                        headers = table_rows[0] if table_rows else []
                        rows = table_rows[1:] if len(table_rows) > 1 else []
                        
                        # Calculate average confidence
                        all_confidences = []
                        for row_data in lines_by_y.values():
                            all_confidences.extend([col['confidence'] for col in row_data])
                        
                        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
                        
                        tables.append(TableData(
                            headers=headers,
                            rows=rows,
                            confidence=avg_confidence
                        ))
        except Exception as e:
            logger.error(f"Error extracting table data: {str(e)}")
        
        return tables
    
    def _detect_language(self, text: str, hint: Optional[str] = None) -> str:
        """Detect document language."""
        if hint and hint in self._supported_languages:
            return hint
        
        # Simple language detection based on character patterns
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'ch'  # Chinese
        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return 'ja'  # Japanese
        elif re.search(r'[\uac00-\ud7af]', text):
            return 'ko'  # Korean
        elif re.search(r'[\u0600-\u06ff]', text):
            return 'ar'  # Arabic
        elif re.search(r'[\u0900-\u097f]', text):
            return 'hi'  # Hindi
        else:
            return self.default_lang  # Default to English
    
    async def extract_structured_data(self, document: ProcessedDocument) -> StructuredData:
        """
        Extract structured data from processed document.
        
        Args:
            document: Processed document from OCR
            
        Returns:
            Structured data extracted from document
        """
        text = document.extracted_text
        doc_type = document.document_type
        
        # Extract key fields based on document type
        key_fields = {}
        dates = []
        financial_info = None
        missing_fields = []
        
        if doc_type == DocumentType.PASSPORT:
            key_fields, dates, missing_fields = self._extract_passport_data(text)
        elif doc_type == DocumentType.BANK_STATEMENT:
            key_fields, dates, financial_info, missing_fields = self._extract_bank_statement_data(text)
        elif doc_type == DocumentType.EMPLOYMENT_LETTER:
            key_fields, dates, missing_fields = self._extract_employment_data(text)
        elif doc_type == DocumentType.EDUCATIONAL_CERTIFICATE:
            key_fields, dates, missing_fields = self._extract_education_data(text)
        elif doc_type == DocumentType.MEDICAL_CERTIFICATE:
            key_fields, dates, missing_fields = self._extract_medical_data(text)
        elif doc_type == DocumentType.VISA_APPLICATION:
            key_fields, dates, missing_fields = self._extract_visa_application_data(text)
        
        return StructuredData(
            document_type=doc_type,
            key_fields=key_fields,
            dates=dates,
            financial_info=financial_info,
            missing_fields=missing_fields,
            extraction_confidence=document.confidence_score
        )
    
    def _extract_passport_data(self, text: str) -> Tuple[Dict[str, Any], List[DateField], List[str]]:
        """Extract structured data from passport."""
        key_fields = {}
        dates = []
        missing_fields = []
        
        # Extract passport number
        passport_match = re.search(r'(?:passport\s+no\.?|passport\s+number)\s*:?\s*([A-Z0-9]+)', text, re.IGNORECASE)
        if passport_match:
            key_fields['passport_number'] = passport_match.group(1)
        else:
            missing_fields.append('passport_number')
        
        # Extract name
        name_match = re.search(r'(?:name|surname|given\s+names?)\s*:?\s*([A-Z\s]+)', text, re.IGNORECASE)
        if name_match:
            key_fields['name'] = name_match.group(1).strip()
        else:
            missing_fields.append('name')
        
        # Extract nationality
        nationality_match = re.search(r'nationality\s*:?\s*([A-Z\s]+)', text, re.IGNORECASE)
        if nationality_match:
            key_fields['nationality'] = nationality_match.group(1).strip()
        else:
            missing_fields.append('nationality')
        
        # Extract dates
        date_patterns = [
            (r'date\s+of\s+birth\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', 'date_of_birth'),
            (r'date\s+of\s+issue\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', 'date_of_issue'),
            (r'date\s+of\s+expiry\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', 'date_of_expiry')
        ]
        
        for pattern, field_name in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1)
                    # Simple date parsing - can be enhanced
                    date_obj = self._parse_date(date_str)
                    dates.append(DateField(
                        field_name=field_name,
                        date_value=date_obj,
                        confidence=0.8,
                        format_detected=date_str
                    ))
                except:
                    missing_fields.append(field_name)
            else:
                missing_fields.append(field_name)
        
        return key_fields, dates, missing_fields
    
    def _extract_bank_statement_data(self, text: str) -> Tuple[Dict[str, Any], List[DateField], Optional[FinancialData], List[str]]:
        """Extract structured data from bank statement."""
        key_fields = {}
        dates = []
        missing_fields = []
        
        # Extract account information
        account_match = re.search(r'account\s+(?:no\.?|number)\s*:?\s*([0-9\-]+)', text, re.IGNORECASE)
        if account_match:
            key_fields['account_number'] = account_match.group(1)
        else:
            missing_fields.append('account_number')
        
        # Extract account holder name
        holder_match = re.search(r'(?:account\s+holder|name)\s*:?\s*([A-Z\s]+)', text, re.IGNORECASE)
        if holder_match:
            key_fields['account_holder'] = holder_match.group(1).strip()
        else:
            missing_fields.append('account_holder')
        
        # Extract financial data
        amounts = []
        currency = None
        
        # Find currency symbols/codes
        currency_match = re.search(r'(\$|€|£|USD|EUR|GBP|INR|CAD)', text)
        if currency_match:
            currency = currency_match.group(1)
        
        # Extract amounts
        amount_pattern = r'(?:\$|€|£|USD|EUR|GBP|INR|CAD)?\s*([0-9,]+\.?[0-9]*)'
        amount_matches = re.findall(amount_pattern, text)
        
        for amount_str in amount_matches:
            try:
                amount = float(amount_str.replace(',', ''))
                amounts.append({'amount': amount, 'currency': currency})
            except:
                continue
        
        financial_info = FinancialData(
            amounts=amounts,
            currency=currency,
            account_numbers=[key_fields.get('account_number', '')],
            transaction_count=len(amounts)
        ) if amounts else None
        
        # Extract statement period dates
        period_match = re.search(r'statement\s+period\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\s*(?:to|-)?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', text, re.IGNORECASE)
        if period_match:
            try:
                start_date = self._parse_date(period_match.group(1))
                end_date = self._parse_date(period_match.group(2))
                dates.extend([
                    DateField(field_name='statement_start', date_value=start_date, confidence=0.8, format_detected=period_match.group(1)),
                    DateField(field_name='statement_end', date_value=end_date, confidence=0.8, format_detected=period_match.group(2))
                ])
            except:
                missing_fields.extend(['statement_start', 'statement_end'])
        else:
            missing_fields.extend(['statement_start', 'statement_end'])
        
        return key_fields, dates, financial_info, missing_fields
    
    def _extract_employment_data(self, text: str) -> Tuple[Dict[str, Any], List[DateField], List[str]]:
        """Extract structured data from employment letter."""
        key_fields = {}
        dates = []
        missing_fields = []
        
        # Extract employee name
        name_match = re.search(r'(?:employee|mr\.?|ms\.?|mrs\.?)\s+([A-Z\s]+)', text, re.IGNORECASE)
        if name_match:
            key_fields['employee_name'] = name_match.group(1).strip()
        else:
            missing_fields.append('employee_name')
        
        # Extract position/job title
        position_match = re.search(r'(?:position|job\s+title|role)\s*:?\s*([A-Z\s]+)', text, re.IGNORECASE)
        if position_match:
            key_fields['position'] = position_match.group(1).strip()
        else:
            missing_fields.append('position')
        
        # Extract salary
        salary_match = re.search(r'salary\s*:?\s*(?:\$|€|£|USD|EUR|GBP)?\s*([0-9,]+\.?[0-9]*)', text, re.IGNORECASE)
        if salary_match:
            key_fields['salary'] = salary_match.group(1)
        else:
            missing_fields.append('salary')
        
        # Extract employment dates
        start_date_match = re.search(r'(?:start\s+date|employment\s+from)\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', text, re.IGNORECASE)
        if start_date_match:
            try:
                start_date = self._parse_date(start_date_match.group(1))
                dates.append(DateField(
                    field_name='employment_start',
                    date_value=start_date,
                    confidence=0.8,
                    format_detected=start_date_match.group(1)
                ))
            except:
                missing_fields.append('employment_start')
        else:
            missing_fields.append('employment_start')
        
        return key_fields, dates, missing_fields
    
    def _extract_education_data(self, text: str) -> Tuple[Dict[str, Any], List[DateField], List[str]]:
        """Extract structured data from educational certificate."""
        key_fields = {}
        dates = []
        missing_fields = []
        
        # Extract student name
        name_match = re.search(r'(?:this\s+is\s+to\s+certify\s+that|student\s+name)\s+([A-Z\s]+)', text, re.IGNORECASE)
        if name_match:
            key_fields['student_name'] = name_match.group(1).strip()
        else:
            missing_fields.append('student_name')
        
        # Extract degree/qualification
        degree_match = re.search(r'(?:degree|diploma|certificate)\s+(?:of|in)\s+([A-Z\s]+)', text, re.IGNORECASE)
        if degree_match:
            key_fields['degree'] = degree_match.group(1).strip()
        else:
            missing_fields.append('degree')
        
        # Extract institution
        institution_match = re.search(r'(?:university|college|institution)\s+(?:of|name)\s*:?\s*([A-Z\s]+)', text, re.IGNORECASE)
        if institution_match:
            key_fields['institution'] = institution_match.group(1).strip()
        else:
            missing_fields.append('institution')
        
        # Extract graduation date
        grad_date_match = re.search(r'(?:graduation\s+date|completed\s+on)\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', text, re.IGNORECASE)
        if grad_date_match:
            try:
                grad_date = self._parse_date(grad_date_match.group(1))
                dates.append(DateField(
                    field_name='graduation_date',
                    date_value=grad_date,
                    confidence=0.8,
                    format_detected=grad_date_match.group(1)
                ))
            except:
                missing_fields.append('graduation_date')
        else:
            missing_fields.append('graduation_date')
        
        return key_fields, dates, missing_fields
    
    def _extract_medical_data(self, text: str) -> Tuple[Dict[str, Any], List[DateField], List[str]]:
        """Extract structured data from medical certificate."""
        key_fields = {}
        dates = []
        missing_fields = []
        
        # Extract patient name
        name_match = re.search(r'(?:patient\s+name|name)\s*:?\s*([A-Z\s]+)', text, re.IGNORECASE)
        if name_match:
            key_fields['patient_name'] = name_match.group(1).strip()
        else:
            missing_fields.append('patient_name')
        
        # Extract doctor name
        doctor_match = re.search(r'(?:doctor|physician|dr\.)\s+([A-Z\s]+)', text, re.IGNORECASE)
        if doctor_match:
            key_fields['doctor_name'] = doctor_match.group(1).strip()
        else:
            missing_fields.append('doctor_name')
        
        # Extract examination date
        exam_date_match = re.search(r'(?:examination\s+date|date\s+of\s+examination)\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', text, re.IGNORECASE)
        if exam_date_match:
            try:
                exam_date = self._parse_date(exam_date_match.group(1))
                dates.append(DateField(
                    field_name='examination_date',
                    date_value=exam_date,
                    confidence=0.8,
                    format_detected=exam_date_match.group(1)
                ))
            except:
                missing_fields.append('examination_date')
        else:
            missing_fields.append('examination_date')
        
        return key_fields, dates, missing_fields
    
    def _extract_visa_application_data(self, text: str) -> Tuple[Dict[str, Any], List[DateField], List[str]]:
        """Extract structured data from visa application form."""
        key_fields = {}
        dates = []
        missing_fields = []
        
        # Extract applicant name
        name_match = re.search(r'(?:applicant\s+name|full\s+name)\s*:?\s*([A-Z\s]+)', text, re.IGNORECASE)
        if name_match:
            key_fields['applicant_name'] = name_match.group(1).strip()
        else:
            missing_fields.append('applicant_name')
        
        # Extract visa type
        visa_type_match = re.search(r'(?:visa\s+type|type\s+of\s+visa)\s*:?\s*([A-Z0-9\s]+)', text, re.IGNORECASE)
        if visa_type_match:
            key_fields['visa_type'] = visa_type_match.group(1).strip()
        else:
            missing_fields.append('visa_type')
        
        # Extract intended travel date
        travel_date_match = re.search(r'(?:intended\s+travel\s+date|departure\s+date)\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', text, re.IGNORECASE)
        if travel_date_match:
            try:
                travel_date = self._parse_date(travel_date_match.group(1))
                dates.append(DateField(
                    field_name='intended_travel_date',
                    date_value=travel_date,
                    confidence=0.8,
                    format_detected=travel_date_match.group(1)
                ))
            except:
                missing_fields.append('intended_travel_date')
        else:
            missing_fields.append('intended_travel_date')
        
        return key_fields, dates, missing_fields
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string into datetime object."""
        # Remove extra whitespace
        date_str = date_str.strip()
        
        # Common date formats
        formats = [
            '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
            '%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d',
            '%d.%m.%Y', '%m.%d.%Y', '%Y.%m.%d',
            '%d/%m/%y', '%m/%d/%y', '%y/%m/%d',
            '%d-%m-%y', '%m-%d-%y', '%y-%m-%d'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # If no format matches, raise error
        raise ValueError(f"Unable to parse date: {date_str}")
    
    async def validate_document_quality(self, document: ProcessedDocument) -> QualityScore:
        """
        Assess the quality of document processing results.
        
        Args:
            document: Processed document to assess
            
        Returns:
            Quality score and assessment details
        """
        issues = []
        
        # Text clarity assessment
        text_clarity = document.confidence_score
        if text_clarity < 0.5:
            issues.append("Low OCR confidence - text may be unclear")
        
        # Image quality assessment (based on confidence and text length)
        image_quality = min(1.0, len(document.extracted_text) / 100)  # Normalize by expected text length
        if image_quality < 0.3:
            issues.append("Very short extracted text - image quality may be poor")
        
        # Completeness assessment
        completeness = 1.0
        if not document.extracted_text.strip():
            completeness = 0.0
            issues.append("No text extracted from document")
        elif len(document.extracted_text) < 50:
            completeness = 0.5
            issues.append("Very little text extracted - document may be incomplete")
        
        # Overall score
        overall_score = (text_clarity + image_quality + completeness) / 3
        
        return QualityScore(
            overall_score=overall_score,
            text_clarity=text_clarity,
            image_quality=image_quality,
            completeness=completeness,
            issues=issues
        )