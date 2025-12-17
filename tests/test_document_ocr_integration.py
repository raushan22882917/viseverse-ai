"""
Unit tests for OCR integration in Document Processing Service.
Tests specific document types, error handling, and multilingual processing.

Requirements: 1.1, 4.1, 4.2
"""

import asyncio
import io
import pytest
from PIL import Image, ImageDraw, ImageFont
from unittest.mock import Mock, patch, AsyncMock

from src.visaverse.services.document.processor import PaddleOCRDocumentProcessor
from src.visaverse.core.models import (
    DocumentType, 
    ProcessingStatus, 
    StructuredData,
    QualityScore
)


class TestOCRIntegration:
    """Test OCR integration functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create document processor instance."""
        return PaddleOCRDocumentProcessor(use_gpu=False, lang='en')
    
    def create_test_image(self, text_content: str, width: int = 400, height: int = 300) -> bytes:
        """Create a test image with specified text content."""
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        # Split text into lines and draw
        lines = text_content.split('\n')
        y_offset = 50
        
        for line in lines:
            if line.strip():
                draw.text((50, y_offset), line, fill='black', font=font)
                y_offset += 25
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
    
    def test_passport_document_processing(self, processor):
        """Test processing of passport documents."""
        async def run_test():
            # Create passport-like content
            passport_text = """PASSPORT
Passport No: AB123456789
Name: JOHN MICHAEL DOE
Nationality: AMERICAN
Date of Birth: 15/03/1985
Date of Issue: 10/01/2020
Date of Expiry: 10/01/2030
Place of Birth: NEW YORK"""
            
            image_bytes = self.create_test_image(passport_text)
            
            # Process document
            processed_docs = await processor.process_documents(
                [image_bytes], 
                ['passport.png']
            )
            
            assert len(processed_docs) == 1
            doc = processed_docs[0]
            
            # Should classify as passport
            assert doc.document_type == DocumentType.PASSPORT
            assert doc.processing_status == ProcessingStatus.COMPLETED
            
            # Should extract text
            assert len(doc.extracted_text) > 0
            assert 'passport' in doc.extracted_text.lower()
            
            # Should have confidence score
            assert 0.0 <= doc.confidence_score <= 1.0
            
            # Extract structured data
            structured_data = await processor.extract_structured_data(doc)
            
            # Should extract passport-specific fields
            assert structured_data.document_type == DocumentType.PASSPORT
            assert isinstance(structured_data.key_fields, dict)
            
            # Should identify missing fields if any
            assert isinstance(structured_data.missing_fields, list)
        
        asyncio.run(run_test())
    
    def test_bank_statement_document_processing(self, processor):
        """Test processing of bank statement documents."""
        async def run_test():
            # Create bank statement-like content
            bank_text = """BANK STATEMENT
Account Number: 1234567890
Account Holder: JANE SMITH
Statement Period: 01/01/2024 to 31/01/2024
Opening Balance: $5,000.00
Closing Balance: $4,750.00
Transaction 1: -$250.00 Grocery Store
Transaction 2: +$1,000.00 Salary Deposit"""
            
            image_bytes = self.create_test_image(bank_text)
            
            # Process document
            processed_docs = await processor.process_documents(
                [image_bytes], 
                ['bank_statement.pdf']
            )
            
            assert len(processed_docs) == 1
            doc = processed_docs[0]
            
            # Should classify as bank statement
            assert doc.document_type == DocumentType.BANK_STATEMENT
            assert doc.processing_status == ProcessingStatus.COMPLETED
            
            # Extract structured data
            structured_data = await processor.extract_structured_data(doc)
            
            # Should extract financial information
            assert structured_data.document_type == DocumentType.BANK_STATEMENT
            
            # Should have financial data if extracted properly
            if structured_data.financial_info:
                assert isinstance(structured_data.financial_info.amounts, list)
                assert structured_data.financial_info.currency is not None
        
        asyncio.run(run_test())
    
    def test_visa_application_document_processing(self, processor):
        """Test processing of visa application documents."""
        async def run_test():
            # Create visa application-like content
            visa_text = """VISA APPLICATION FORM
Applicant Name: MARIA GONZALEZ
Visa Type: Tourist
Intended Travel Date: 15/06/2024
Purpose of Visit: Tourism
Duration of Stay: 14 days
Passport Number: CD987654321"""
            
            image_bytes = self.create_test_image(visa_text)
            
            # Process document
            processed_docs = await processor.process_documents(
                [image_bytes], 
                ['visa_application.pdf']
            )
            
            assert len(processed_docs) == 1
            doc = processed_docs[0]
            
            # Should classify as visa application
            assert doc.document_type == DocumentType.VISA_APPLICATION
            assert doc.processing_status == ProcessingStatus.COMPLETED
            
            # Extract structured data
            structured_data = await processor.extract_structured_data(doc)
            
            # Should extract visa application fields
            assert structured_data.document_type == DocumentType.VISA_APPLICATION
            assert isinstance(structured_data.key_fields, dict)
        
        asyncio.run(run_test())
    
    def test_corrupted_file_handling(self, processor):
        """Test error handling for corrupted files."""
        async def run_test():
            # Create invalid image data
            corrupted_data = b"This is not a valid image file"
            
            # Should handle corrupted file gracefully
            processed_docs = await processor.process_documents(
                [corrupted_data], 
                ['corrupted.png']
            )
            
            assert len(processed_docs) == 1
            doc = processed_docs[0]
            
            # Should mark as failed
            assert doc.processing_status == ProcessingStatus.FAILED
            
            # Should still have basic structure
            assert isinstance(doc.document_type, DocumentType)
            assert doc.confidence_score == 0.0
        
        asyncio.run(run_test())
    
    def test_empty_image_handling(self, processor):
        """Test handling of empty or blank images."""
        async def run_test():
            # Create blank image
            blank_image = Image.new('RGB', (100, 100), color='white')
            img_byte_arr = io.BytesIO()
            blank_image.save(img_byte_arr, format='PNG')
            blank_bytes = img_byte_arr.getvalue()
            
            # Process blank image
            processed_docs = await processor.process_documents(
                [blank_bytes], 
                ['blank.png']
            )
            
            assert len(processed_docs) == 1
            doc = processed_docs[0]
            
            # Should complete processing even if no text found
            assert doc.processing_status == ProcessingStatus.COMPLETED
            
            # Should have low confidence or empty text
            assert len(doc.extracted_text.strip()) == 0 or doc.confidence_score < 0.5
            
            # Quality assessment should reflect poor quality
            quality_score = await processor.validate_document_quality(doc)
            assert quality_score.overall_score < 0.5
            assert len(quality_score.issues) > 0
        
        asyncio.run(run_test())
    
    def test_multilingual_document_processing(self, processor):
        """Test multilingual document processing."""
        async def run_test():
            # Test with different language hints
            test_text = "PASSPORT\nName: JOHN DOE\nNationality: AMERICAN"
            image_bytes = self.create_test_image(test_text)
            
            # Test with English
            processed_docs_en = await processor.process_documents(
                [image_bytes], 
                ['passport_en.png'], 
                language='en'
            )
            
            assert len(processed_docs_en) == 1
            doc_en = processed_docs_en[0]
            assert doc_en.language == 'en'
            
            # Test with French hint
            processed_docs_fr = await processor.process_documents(
                [image_bytes], 
                ['passport_fr.png'], 
                language='fr'
            )
            
            assert len(processed_docs_fr) == 1
            doc_fr = processed_docs_fr[0]
            assert doc_fr.language == 'fr'
            
            # Test with unsupported language (should fall back)
            processed_docs_unsupported = await processor.process_documents(
                [image_bytes], 
                ['passport_unsupported.png'], 
                language='xyz'
            )
            
            assert len(processed_docs_unsupported) == 1
            doc_unsupported = processed_docs_unsupported[0]
            # Should fall back to supported language
            assert doc_unsupported.language in ['en', 'ch', 'ta', 'te', 'ka', 'ja', 'ko', 'hi', 'ar', 'fr', 'de', 'es', 'pt', 'ru', 'it']
        
        asyncio.run(run_test())
    
    def test_document_type_classification(self, processor):
        """Test document type classification accuracy."""
        async def run_test():
            # Test different document types
            test_cases = [
                ("EMPLOYMENT LETTER\nEmployee: John Doe\nPosition: Engineer\nSalary: $75000", 
                 "employment_letter.pdf", DocumentType.EMPLOYMENT_LETTER),
                ("MEDICAL CERTIFICATE\nPatient: Jane Smith\nDoctor: Dr. Brown\nExamination Date: 01/01/2024", 
                 "medical_cert.pdf", DocumentType.MEDICAL_CERTIFICATE),
                ("CERTIFICATE\nThis certifies that John Doe\nhas completed Bachelor of Science\nUniversity of Example", 
                 "diploma.pdf", DocumentType.EDUCATIONAL_CERTIFICATE),
            ]
            
            for text_content, filename, expected_type in test_cases:
                image_bytes = self.create_test_image(text_content)
                
                processed_docs = await processor.process_documents(
                    [image_bytes], 
                    [filename]
                )
                
                assert len(processed_docs) == 1
                doc = processed_docs[0]
                
                # Should classify correctly based on content and filename
                assert doc.document_type == expected_type
                assert doc.processing_status == ProcessingStatus.COMPLETED
        
        asyncio.run(run_test())
    
    def test_quality_assessment(self, processor):
        """Test document quality assessment functionality."""
        async def run_test():
            # Create high-quality document
            high_quality_text = """PASSPORT
Passport Number: AB123456789
Full Name: JOHN MICHAEL DOE
Nationality: UNITED STATES OF AMERICA
Date of Birth: 15 March 1985
Date of Issue: 10 January 2020
Date of Expiry: 10 January 2030
Place of Birth: New York, NY
Sex: M"""
            
            high_quality_image = self.create_test_image(high_quality_text, 800, 600)
            
            processed_docs = await processor.process_documents(
                [high_quality_image], 
                ['high_quality_passport.png']
            )
            
            assert len(processed_docs) == 1
            doc = processed_docs[0]
            
            # Assess quality
            quality_score = await processor.validate_document_quality(doc)
            
            # Should have quality metrics
            assert 0.0 <= quality_score.overall_score <= 1.0
            assert 0.0 <= quality_score.text_clarity <= 1.0
            assert 0.0 <= quality_score.image_quality <= 1.0
            assert 0.0 <= quality_score.completeness <= 1.0
            
            # Should provide issues list
            assert isinstance(quality_score.issues, list)
            
            # For good quality document, should have reasonable scores
            if doc.processing_status == ProcessingStatus.COMPLETED and len(doc.extracted_text) > 100:
                assert quality_score.overall_score > 0.3  # Should be reasonably good
        
        asyncio.run(run_test())
    
    def test_structured_data_extraction_completeness(self, processor):
        """Test that structured data extraction covers all required fields."""
        async def run_test():
            # Test passport data extraction
            passport_text = """PASSPORT
Passport No: XY987654321
Name: ALICE WONDERLAND
Nationality: BRITISH
Date of Birth: 25/12/1990
Date of Issue: 15/06/2022
Date of Expiry: 15/06/2032"""
            
            image_bytes = self.create_test_image(passport_text)
            
            processed_docs = await processor.process_documents(
                [image_bytes], 
                ['passport_complete.png']
            )
            
            doc = processed_docs[0]
            structured_data = await processor.extract_structured_data(doc)
            
            # Should extract key passport fields
            assert structured_data.document_type == DocumentType.PASSPORT
            assert isinstance(structured_data.key_fields, dict)
            assert isinstance(structured_data.dates, list)
            assert isinstance(structured_data.missing_fields, list)
            
            # Should have extraction confidence
            assert 0.0 <= structured_data.extraction_confidence <= 1.0
            
            # For passport, should attempt to extract standard fields
            expected_fields = ['passport_number', 'name', 'nationality']
            expected_dates = ['date_of_birth', 'date_of_issue', 'date_of_expiry']
            
            # Check if fields were extracted or marked as missing
            for field in expected_fields:
                assert field in structured_data.key_fields or field in structured_data.missing_fields
            
            for date_field in expected_dates:
                date_found = any(date.field_name == date_field for date in structured_data.dates)
                assert date_found or date_field in structured_data.missing_fields
        
        asyncio.run(run_test())
    
    def test_table_data_extraction(self, processor):
        """Test extraction of table data from documents."""
        async def run_test():
            # Create document with table-like structure
            table_text = """BANK STATEMENT
Account: 123456789
Date        Description         Amount
01/01/2024  Opening Balance     $1000.00
05/01/2024  Grocery Store       -$50.00
10/01/2024  Salary Deposit      +$2000.00
15/01/2024  Utility Bill        -$150.00"""
            
            image_bytes = self.create_test_image(table_text, 600, 400)
            
            processed_docs = await processor.process_documents(
                [image_bytes], 
                ['statement_with_table.pdf']
            )
            
            doc = processed_docs[0]
            
            # Should extract table data
            assert isinstance(doc.table_data, list)
            
            # If table detected, should have structure
            if len(doc.table_data) > 0:
                table = doc.table_data[0]
                assert isinstance(table.headers, list)
                assert isinstance(table.rows, list)
                assert 0.0 <= table.confidence <= 1.0
        
        asyncio.run(run_test())
    
    def test_date_parsing_robustness(self, processor):
        """Test robustness of date parsing functionality."""
        async def run_test():
            # Test various date formats
            date_formats_text = """DOCUMENT
Date 1: 15/03/1985
Date 2: 03-15-1985
Date 3: 1985.03.15
Date 4: 15/3/85
Date 5: 3-15-85"""
            
            image_bytes = self.create_test_image(date_formats_text)
            
            processed_docs = await processor.process_documents(
                [image_bytes], 
                ['date_formats.png']
            )
            
            doc = processed_docs[0]
            structured_data = await processor.extract_structured_data(doc)
            
            # Should handle various date formats
            assert isinstance(structured_data.dates, list)
            
            # Each extracted date should have required fields
            for date_field in structured_data.dates:
                assert hasattr(date_field, 'field_name')
                assert hasattr(date_field, 'date_value')
                assert hasattr(date_field, 'confidence')
                assert hasattr(date_field, 'format_detected')
                assert 0.0 <= date_field.confidence <= 1.0
        
        asyncio.run(run_test())


# Additional integration tests for error scenarios
class TestOCRErrorHandling:
    """Test error handling in OCR integration."""
    
    @pytest.fixture
    def processor(self):
        """Create document processor instance."""
        return PaddleOCRDocumentProcessor(use_gpu=False, lang='en')
    
    def test_mismatched_files_and_filenames(self, processor):
        """Test error handling when files and filenames don't match."""
        async def run_test():
            image = Image.new('RGB', (100, 100), color='white')
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            
            files = [img_byte_arr.getvalue()]
            filenames = ['file1.png', 'file2.png']  # Mismatch
            
            # Should raise ValueError
            with pytest.raises(ValueError, match="Number of files must match number of filenames"):
                await processor.process_documents(files, filenames)
        
        asyncio.run(run_test())
    
    def test_large_document_handling(self, processor):
        """Test handling of large documents."""
        async def run_test():
            # Create large image
            large_text = "LARGE DOCUMENT\n" + "Line of text\n" * 100
            
            image = Image.new('RGB', (1200, 1600), color='white')
            draw = ImageDraw.Draw(image)
            
            lines = large_text.split('\n')
            y_offset = 50
            
            for i, line in enumerate(lines[:50]):  # Limit to avoid too large image
                if line.strip():
                    draw.text((50, y_offset), line, fill='black')
                    y_offset += 20
            
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            
            # Should handle large document
            processed_docs = await processor.process_documents(
                [img_byte_arr.getvalue()], 
                ['large_document.png']
            )
            
            assert len(processed_docs) == 1
            doc = processed_docs[0]
            
            # Should complete processing
            assert doc.processing_status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]
        
        asyncio.run(run_test())


# Synchronous test wrappers for pytest
def test_passport_processing():
    """Test passport document processing."""
    processor = PaddleOCRDocumentProcessor(use_gpu=False, lang='en')
    test_instance = TestOCRIntegration()
    test_instance.test_passport_document_processing(processor)


def test_bank_statement_processing():
    """Test bank statement processing."""
    processor = PaddleOCRDocumentProcessor(use_gpu=False, lang='en')
    test_instance = TestOCRIntegration()
    test_instance.test_bank_statement_document_processing(processor)


def test_corrupted_file_handling():
    """Test corrupted file handling."""
    processor = PaddleOCRDocumentProcessor(use_gpu=False, lang='en')
    test_instance = TestOCRIntegration()
    test_instance.test_corrupted_file_handling(processor)


def test_multilingual_processing():
    """Test multilingual document processing."""
    processor = PaddleOCRDocumentProcessor(use_gpu=False, lang='en')
    test_instance = TestOCRIntegration()
    test_instance.test_multilingual_document_processing(processor)


def test_document_classification():
    """Test document type classification."""
    processor = PaddleOCRDocumentProcessor(use_gpu=False, lang='en')
    test_instance = TestOCRIntegration()
    test_instance.test_document_type_classification(processor)


def test_quality_assessment():
    """Test document quality assessment."""
    processor = PaddleOCRDocumentProcessor(use_gpu=False, lang='en')
    test_instance = TestOCRIntegration()
    test_instance.test_quality_assessment(processor)


def test_error_handling():
    """Test error handling scenarios."""
    processor = PaddleOCRDocumentProcessor(use_gpu=False, lang='en')
    test_instance = TestOCRErrorHandling()
    test_instance.test_mismatched_files_and_filenames(processor)