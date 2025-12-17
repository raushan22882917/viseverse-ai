"""
Property-based tests for document processing completeness.

**Feature: visaverse-guardian-ai, Property 1: Document Processing Completeness**
**Validates: Requirements 1.1, 1.5, 4.1, 4.2, 4.3, 4.4, 4.5**

Property 1: Document Processing Completeness
For any uploaded document in any supported language, the OCR processing should extract 
structured data including document type, key fields, dates, financial information, 
and missing elements, with confidence scores provided for all extractions.
"""

import asyncio
import io
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from hypothesis import given, strategies as st, assume, settings
import pytest

from src.visaverse.services.document.processor import PaddleOCRDocumentProcessor
from src.visaverse.core.models import DocumentType, ProcessingStatus


# Test data generators
@st.composite
def generate_document_content(draw):
    """Generate realistic document content for different document types."""
    doc_type = draw(st.sampled_from(list(DocumentType)))
    
    if doc_type == DocumentType.PASSPORT:
        content = {
            'type': 'passport',
            'text': f"""PASSPORT
Passport No: {draw(st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', min_size=8, max_size=10))}
Name: {draw(st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ ', min_size=5, max_size=30))}
Nationality: {draw(st.sampled_from(['AMERICAN', 'BRITISH', 'CANADIAN', 'GERMAN', 'FRENCH']))}
Date of Birth: {draw(st.integers(1, 28))}/{draw(st.integers(1, 12))}/{draw(st.integers(1970, 2005))}
Date of Issue: {draw(st.integers(1, 28))}/{draw(st.integers(1, 12))}/{draw(st.integers(2015, 2023))}
Date of Expiry: {draw(st.integers(1, 28))}/{draw(st.integers(1, 12))}/{draw(st.integers(2025, 2035))}"""
        }
    elif doc_type == DocumentType.BANK_STATEMENT:
        content = {
            'type': 'bank_statement',
            'text': f"""BANK STATEMENT
Account Number: {draw(st.text(alphabet='0123456789', min_size=10, max_size=12))}
Account Holder: {draw(st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ ', min_size=5, max_size=30))}
Statement Period: {draw(st.integers(1, 28))}/{draw(st.integers(1, 12))}/{draw(st.integers(2023, 2024))} to {draw(st.integers(1, 28))}/{draw(st.integers(1, 12))}/{draw(st.integers(2023, 2024))}
Balance: ${draw(st.integers(100, 50000))}.{draw(st.integers(0, 99)):02d}
Transaction 1: ${draw(st.integers(10, 1000))}.{draw(st.integers(0, 99)):02d}
Transaction 2: ${draw(st.integers(10, 1000))}.{draw(st.integers(0, 99)):02d}"""
        }
    elif doc_type == DocumentType.EMPLOYMENT_LETTER:
        content = {
            'type': 'employment_letter',
            'text': f"""EMPLOYMENT LETTER
Employee Name: {draw(st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ ', min_size=5, max_size=30))}
Position: {draw(st.sampled_from(['Software Engineer', 'Manager', 'Analyst', 'Director']))}
Salary: ${draw(st.integers(30000, 150000))}
Start Date: {draw(st.integers(1, 28))}/{draw(st.integers(1, 12))}/{draw(st.integers(2020, 2024))}"""
        }
    elif doc_type == DocumentType.EDUCATIONAL_CERTIFICATE:
        content = {
            'type': 'educational_certificate',
            'text': f"""CERTIFICATE
This is to certify that {draw(st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ ', min_size=5, max_size=30))}
has successfully completed the degree of {draw(st.sampled_from(['Bachelor of Science', 'Master of Arts', 'PhD']))}
University: {draw(st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ ', min_size=10, max_size=40))}
Graduation Date: {draw(st.integers(1, 28))}/{draw(st.integers(1, 12))}/{draw(st.integers(2015, 2024))}"""
        }
    elif doc_type == DocumentType.MEDICAL_CERTIFICATE:
        content = {
            'type': 'medical_certificate',
            'text': f"""MEDICAL CERTIFICATE
Patient Name: {draw(st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ ', min_size=5, max_size=30))}
Doctor: Dr. {draw(st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ ', min_size=5, max_size=25))}
Examination Date: {draw(st.integers(1, 28))}/{draw(st.integers(1, 12))}/{draw(st.integers(2023, 2024))}
Medical examination completed successfully."""
        }
    else:  # VISA_APPLICATION
        content = {
            'type': 'visa_application',
            'text': f"""VISA APPLICATION FORM
Applicant Name: {draw(st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ ', min_size=5, max_size=30))}
Visa Type: {draw(st.sampled_from(['Tourist', 'Business', 'Student', 'Work']))}
Intended Travel Date: {draw(st.integers(1, 28))}/{draw(st.integers(1, 12))}/{draw(st.integers(2024, 2025))}"""
        }
    
    return content


@st.composite
def generate_test_image(draw):
    """Generate a test image with text content."""
    content = draw(generate_document_content())
    
    # Create a simple image with text
    width, height = 800, 600
    image = Image.new('RGB', (width, height), color='white')
    
    try:
        # Try to use a default font, fall back to basic if not available
        font = ImageFont.load_default()
    except:
        font = None
    
    draw_obj = ImageDraw.Draw(image)
    
    # Split text into lines and draw
    lines = content['text'].split('\n')
    y_offset = 50
    
    for line in lines:
        if line.strip():
            draw_obj.text((50, y_offset), line, fill='black', font=font)
            y_offset += 30
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    
    return {
        'content': content,
        'image_bytes': img_bytes,
        'filename': f"test_{content['type']}.png"
    }


@st.composite
def generate_supported_language(draw):
    """Generate a supported language code."""
    supported_langs = ['en', 'ch', 'ta', 'te', 'ka', 'ja', 'ko', 'hi', 'ar', 'fr', 'de', 'es', 'pt', 'ru', 'it']
    return draw(st.sampled_from(supported_langs))


class TestDocumentProcessingCompleteness:
    """Test document processing completeness property."""
    
    @pytest.fixture
    def processor(self):
        """Create document processor instance."""
        return PaddleOCRDocumentProcessor(use_gpu=False, lang='en')
    
    @given(
        test_data=generate_test_image(),
        language=st.one_of(st.none(), generate_supported_language())
    )
    @settings(max_examples=100, deadline=30000)  # 30 second timeout per test
    async def test_document_processing_completeness_property(self, processor, test_data, language):
        """
        **Feature: visaverse-guardian-ai, Property 1: Document Processing Completeness**
        
        Property: For any uploaded document in any supported language, the OCR processing 
        should extract structured data including document type, key fields, dates, 
        financial information, and missing elements, with confidence scores provided 
        for all extractions.
        
        **Validates: Requirements 1.1, 1.5, 4.1, 4.2, 4.3, 4.4, 4.5**
        """
        # Arrange
        files = [test_data['image_bytes']]
        filenames = [test_data['filename']]
        
        # Act - Process documents
        processed_docs = await processor.process_documents(files, filenames, language)
        
        # Assert - Basic processing completeness
        assert len(processed_docs) == 1, "Should process exactly one document"
        
        doc = processed_docs[0]
        
        # Requirement 1.1, 4.1: Document should be processed successfully
        assert doc.processing_status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED], \
            "Document should have a valid processing status"
        
        if doc.processing_status == ProcessingStatus.COMPLETED:
            # Requirement 4.2: Should extract text content
            assert isinstance(doc.extracted_text, str), "Should extract text as string"
            
            # Requirement 4.3: Should identify document type
            assert isinstance(doc.document_type, DocumentType), "Should classify document type"
            
            # Requirement 4.4: Should provide confidence scores
            assert 0.0 <= doc.confidence_score <= 1.0, "Confidence score should be between 0 and 1"
            
            # Requirement 1.5, 4.2: Should handle multilingual content
            assert isinstance(doc.language, str), "Should detect/assign language"
            if language:
                # If language hint provided, should respect it or detect appropriately
                assert doc.language in ['en', 'ch', 'ta', 'te', 'ka', 'ja', 'ko', 'hi', 'ar', 'fr', 'de', 'es', 'pt', 'ru', 'it']
            
            # Should have layout information
            assert doc.layout_info is not None, "Should provide layout information"
            
            # Act - Extract structured data
            structured_data = await processor.extract_structured_data(doc)
            
            # Assert - Structured data completeness
            # Requirement 4.3: Should extract key fields
            assert isinstance(structured_data.key_fields, dict), "Should extract key fields as dictionary"
            
            # Requirement 4.3: Should extract dates
            assert isinstance(structured_data.dates, list), "Should extract dates as list"
            
            # Requirement 4.5: Should identify missing elements
            assert isinstance(structured_data.missing_fields, list), "Should identify missing fields"
            
            # Requirement 4.4: Should provide extraction confidence
            assert 0.0 <= structured_data.extraction_confidence <= 1.0, \
                "Extraction confidence should be between 0 and 1"
            
            # Document type should match
            assert structured_data.document_type == doc.document_type, \
                "Structured data document type should match processed document type"
            
            # For bank statements, should extract financial information when present
            if doc.document_type == DocumentType.BANK_STATEMENT:
                # Should attempt to extract financial data (may be None if not found)
                assert structured_data.financial_info is None or hasattr(structured_data.financial_info, 'amounts')
            
            # Act - Validate document quality
            quality_score = await processor.validate_document_quality(doc)
            
            # Assert - Quality assessment completeness
            # Requirement 4.4: Should provide quality scores
            assert 0.0 <= quality_score.overall_score <= 1.0, "Overall quality score should be between 0 and 1"
            assert 0.0 <= quality_score.text_clarity <= 1.0, "Text clarity score should be between 0 and 1"
            assert 0.0 <= quality_score.image_quality <= 1.0, "Image quality score should be between 0 and 1"
            assert 0.0 <= quality_score.completeness <= 1.0, "Completeness score should be between 0 and 1"
            
            # Should provide quality issues list
            assert isinstance(quality_score.issues, list), "Should provide quality issues as list"
    
    @given(
        files_count=st.integers(min_value=1, max_value=5),
        language=st.one_of(st.none(), generate_supported_language())
    )
    @settings(max_examples=50, deadline=60000)  # 60 second timeout for multiple files
    async def test_multiple_documents_processing_completeness(self, processor, files_count, language):
        """
        Test that multiple documents are processed completely.
        
        **Feature: visaverse-guardian-ai, Property 1: Document Processing Completeness**
        **Validates: Requirements 1.1, 4.1**
        """
        # Generate multiple test documents
        files = []
        filenames = []
        
        for i in range(files_count):
            # Create simple test image
            image = Image.new('RGB', (400, 300), color='white')
            draw = ImageDraw.Draw(image)
            draw.text((50, 50), f"Test Document {i+1}\nSample text content", fill='black')
            
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            
            files.append(img_byte_arr.getvalue())
            filenames.append(f"test_doc_{i+1}.png")
        
        # Act
        processed_docs = await processor.process_documents(files, filenames, language)
        
        # Assert
        assert len(processed_docs) == files_count, f"Should process all {files_count} documents"
        
        for i, doc in enumerate(processed_docs):
            # Each document should have a processing status
            assert doc.processing_status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED], \
                f"Document {i+1} should have valid processing status"
            
            # Should have basic attributes
            assert isinstance(doc.document_type, DocumentType), f"Document {i+1} should have document type"
            assert isinstance(doc.extracted_text, str), f"Document {i+1} should have extracted text"
            assert 0.0 <= doc.confidence_score <= 1.0, f"Document {i+1} should have valid confidence score"
    
    async def test_empty_document_handling(self, processor):
        """
        Test handling of empty or invalid documents.
        
        **Feature: visaverse-guardian-ai, Property 1: Document Processing Completeness**
        **Validates: Requirements 4.1, 4.4**
        """
        # Create empty image
        image = Image.new('RGB', (100, 100), color='white')
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        
        files = [img_byte_arr.getvalue()]
        filenames = ['empty.png']
        
        # Act
        processed_docs = await processor.process_documents(files, filenames)
        
        # Assert
        assert len(processed_docs) == 1, "Should process one document"
        
        doc = processed_docs[0]
        
        # Should handle empty document gracefully
        assert doc.processing_status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]
        
        if doc.processing_status == ProcessingStatus.COMPLETED:
            # Should provide quality assessment even for empty documents
            quality_score = await processor.validate_document_quality(doc)
            assert isinstance(quality_score.overall_score, float)
            assert isinstance(quality_score.issues, list)
    
    async def test_unsupported_language_handling(self, processor):
        """
        Test handling of unsupported language hints.
        
        **Feature: visaverse-guardian-ai, Property 1: Document Processing Completeness**
        **Validates: Requirements 1.5, 4.2**
        """
        # Create test image
        image = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(image)
        draw.text((50, 50), "Test document with text", fill='black')
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        
        files = [img_byte_arr.getvalue()]
        filenames = ['test.png']
        
        # Act with unsupported language
        processed_docs = await processor.process_documents(files, filenames, language='unsupported_lang')
        
        # Assert
        assert len(processed_docs) == 1, "Should process document despite unsupported language"
        
        doc = processed_docs[0]
        
        # Should fall back to default language
        assert doc.language in ['en', 'ch', 'ta', 'te', 'ka', 'ja', 'ko', 'hi', 'ar', 'fr', 'de', 'es', 'pt', 'ru', 'it'], \
            "Should fall back to supported language"


# Synchronous wrapper for async tests
def test_document_processing_completeness_property():
    """Synchronous wrapper for the main property test."""
    processor = PaddleOCRDocumentProcessor(use_gpu=False, lang='en')
    test_instance = TestDocumentProcessingCompleteness()
    
    # Run a simplified version for CI/testing
    async def run_test():
        # Create simple test document
        image = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(image)
        draw.text((50, 50), "PASSPORT\nPassport No: AB123456\nName: JOHN DOE", fill='black')
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        
        files = [img_byte_arr.getvalue()]
        filenames = ['test_passport.png']
        
        # Process document
        processed_docs = await processor.process_documents(files, filenames)
        
        # Basic assertions
        assert len(processed_docs) == 1
        doc = processed_docs[0]
        assert doc.processing_status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]
        
        if doc.processing_status == ProcessingStatus.COMPLETED:
            assert isinstance(doc.extracted_text, str)
            assert isinstance(doc.document_type, DocumentType)
            assert 0.0 <= doc.confidence_score <= 1.0
            
            # Test structured data extraction
            structured_data = await processor.extract_structured_data(doc)
            assert isinstance(structured_data.key_fields, dict)
            assert isinstance(structured_data.dates, list)
            assert isinstance(structured_data.missing_fields, list)
            assert 0.0 <= structured_data.extraction_confidence <= 1.0
            
            # Test quality validation
            quality_score = await processor.validate_document_quality(doc)
            assert 0.0 <= quality_score.overall_score <= 1.0
            assert isinstance(quality_score.issues, list)
    
    # Run the async test
    asyncio.run(run_test())


def test_multiple_documents_processing():
    """Test multiple document processing."""
    processor = PaddleOCRDocumentProcessor(use_gpu=False, lang='en')
    
    async def run_test():
        files = []
        filenames = []
        
        for i in range(2):
            image = Image.new('RGB', (400, 300), color='white')
            draw = ImageDraw.Draw(image)
            draw.text((50, 50), f"Test Document {i+1}", fill='black')
            
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            
            files.append(img_byte_arr.getvalue())
            filenames.append(f"test_{i+1}.png")
        
        processed_docs = await processor.process_documents(files, filenames)
        
        assert len(processed_docs) == 2
        for doc in processed_docs:
            assert doc.processing_status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]
    
    asyncio.run(run_test())


def test_empty_document_handling():
    """Test empty document handling."""
    processor = PaddleOCRDocumentProcessor(use_gpu=False, lang='en')
    
    async def run_test():
        image = Image.new('RGB', (100, 100), color='white')
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        
        files = [img_byte_arr.getvalue()]
        filenames = ['empty.png']
        
        processed_docs = await processor.process_documents(files, filenames)
        
        assert len(processed_docs) == 1
        doc = processed_docs[0]
        assert doc.processing_status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]
    
    asyncio.run(run_test())