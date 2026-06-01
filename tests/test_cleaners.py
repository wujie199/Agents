import pytest
from core.ports.cleaner import DocumentType, CleaningLevel
from document.rag.bridges.composite_cleaner import CompositeCleaner, CleanerAdapter
from document.rag.bridges.cleaners.base_cleaners import (
    WhitespaceCleaner,
    SpecialCharCleaner,
    HtmlCleaner,
    PrivacyCleaner,
    DuplicateCleaner,
    NoiseCleaner,
    LengthFilterCleaner,
)
from document.rag.bridges.cleaner_factory import (
    build_default_cleaner,
    build_html_cleaner,
    build_legal_cleaner,
    build_enterprise_cleaner,
)


class TestBaseCleaners:
    
    def test_whitespace_cleaner(self):
        cleaner = WhitespaceCleaner()
        text = "这是   一段  文本\n\n\n\n多个空格"
        result = cleaner.clean(text)
        assert "   " not in result
        assert "\n\n\n\n" not in result
    
    def test_special_char_cleaner(self):
        cleaner = SpecialCharCleaner(preserve_chinese=True)
        text = "ä¸­ææµè¯\x00\x1fç¹æ®å­ç¬¦"
        result = cleaner.clean(text)
        assert "\x00" not in result
        assert "\x1f" not in result
        assert "ä¸­ææµè¯" in result
    
    def test_html_cleaner(self):
        cleaner = HtmlCleaner(preserve_links=False)
        text = "<html><body><script>alert('xss')</script><p>æ­£æåå®¹</p></body></html>"
        result = cleaner.clean(text)
        assert "<script>" not in result
        assert "<p>" not in result
        assert "æ­£æåå®¹" in result
    
    def test_privacy_cleaner(self):
        cleaner = PrivacyCleaner(mask_phone=True, mask_email=True)
        text = "联系方式：手机13812345678，邮箱test@example.com"
        result = cleaner.clean(text)
        assert "13812345678" not in result
        assert "test@example.com" not in result
        assert "[PHONE]" in result
        assert "[EMAIL]" in result
    
    def test_duplicate_cleaner(self):
        cleaner = DuplicateCleaner()
        text = "第一行\n第一行\n第二行\n第二行\n第三行"
        result = cleaner.clean(text)
        assert "第一行" in result
        assert "第二行" in result
        assert "第三行" in result
    
    def test_noise_cleaner(self):
        cleaner = NoiseCleaner()
        text = "正常内容\n版权所有 2024\n免责声明：仅供参考\nAll Rights Reserved"
        result = cleaner.clean(text)
        assert "版权" not in result
        assert "åè´£å£°æ" not in result
        assert "All Rights Reserved" not in result
        assert "æ­£æåå®¹" in result
    
    def test_length_filter_cleaner(self):
        cleaner = LengthFilterCleaner(min_length=10)
        text = "短文本\n这是一段较长的文本内容\n又一个短段"
        result = cleaner.clean(text)
        assert "短文本" not in result
        assert "è¿æ¯ä¸æ®µè¾é¿çææ¬åå®¹" in result


class TestDomainCleaners:
    
    def test_legal_document_cleaner(self):
        from document.rag.bridges.cleaners.domain_cleaners import LegalDocumentCleaner
        cleaner = LegalDocumentCleaner()
        text = "第一条 本法适用区域。第二条 以下情形除外。"
        result = cleaner.clean(text)
        assert "第一条" in result
        assert "第二条" in result
    
    def test_technical_doc_cleaner(self):
        from document.rag.bridges.cleaners.domain_cleaners import TechnicalDocCleaner
        cleaner = TechnicalDocCleaner()
        text = "```python\nprint('hello')\n```\nå¸¸é MAX_VALUE = 100"
        result = cleaner.clean(text)
        assert "MAX_VALUE" in result
    
    def test_medical_doc_cleaner(self):
        from document.rag.bridges.cleaners.domain_cleaners import MedicalDocCleaner
        cleaner = MedicalDocCleaner()
        text = "门诊号：ABC123\n患者姓名：张三\n年龄：45岁"
        result = cleaner.clean(text)
        assert "[é¨è¯å·]" in result
        assert "[å¹´é¾]" in result
        assert "å¼ ä¸" not in result


class TestCompositeCleaner:
    
    def test_composite_cleaner_basic(self):
        cleaners = [
            HtmlCleaner(),
            WhitespaceCleaner(),
        ]
        composite = CompositeCleaner(cleaners)
        
        text = "<p>  æµè¯  ææ¬  </p>"
        result = composite.clean(text)
        
        assert "<p>" not in result
        assert "  " not in result
        assert "æµè¯" in result
        assert "ææ¬" in result
    
    def test_composite_cleaner_add_remove(self):
        composite = CompositeCleaner([WhitespaceCleaner()])
        assert len(composite.get_cleaner_names()) == 1
        
        composite.add_cleaner(SpecialCharCleaner())
        assert len(composite.get_cleaner_names()) == 2
        
        removed = composite.remove_cleaner("whitespace")
        assert removed is True
        assert len(composite.get_cleaner_names()) == 1
    
    def test_composite_cleaner_batch(self):
        composite = CompositeCleaner([WhitespaceCleaner()])
        texts = ["文本  一", "文本  二", "文本  三"]
        results = composite.clean_batch(texts)
        
        assert len(results) == 3
        for r in results:
            assert "  " not in r


class TestCleanerAdapter:
    
    def test_cleaner_adapter_default(self):
        adapter = CleanerAdapter()
        
        text = "<p>æµè¯ææ¬ 13812345678</p>"
        result = adapter.clean(text)
        
        assert "<p>" not in result
        assert "13812345678" not in result
    
    def test_cleaner_adapter_with_domain(self):
        adapter = CleanerAdapter()
        
        html_cleaner = CompositeCleaner([HtmlCleaner(), WhitespaceCleaner()])
        adapter.register_domain_cleaner(DocumentType.HTML, html_cleaner)
        
        text = "<div>HTMLåå®¹</div>"
        result = adapter.clean(text, doc_type=DocumentType.HTML)
        
        assert "<div>" not in result
        assert "HTMLåå®¹" in result
    
    def test_cleaner_adapter_batch(self):
        adapter = CleanerAdapter()
        texts = ["<p>一</p>", "<p>二</p>", "<p>三</p>"]
        results = adapter.clean_batch(texts)
        
        assert len(results) == 3
        for r in results:
            assert "<p>" not in r
    
    def test_cleaner_adapter_config_driven(self):
        config = {
            "cleaners": [
                {"type": "whitespace"},
                {"type": "special_char", "preserve_chinese": True},
            ]
        }
        
        adapter = CleanerAdapter()
        cleaner = adapter.build_cleaner_from_config(config)
        
        text = "æµè¯  ææ¬\x00"
        result = cleaner.clean(text)
        
        assert "  " not in result
        assert "\x00" not in result


class TestCleanerFactory:
    
    def test_build_default_cleaner(self):
        cleaner = build_default_cleaner()
        assert cleaner is not None
        assert len(cleaner.get_cleaner_names()) > 0
        
        text = "<p>æµè¯ 13812345678</p>"
        result = cleaner.clean(text)
        assert "<p>" not in result
    
    def test_build_html_cleaner(self):
        cleaner = build_html_cleaner()
        
        html = "<html><body><nav>导航栏</nav><p>这是正常内容段落</p></body></html>"
        result = cleaner.clean(html)
        
        assert "<html>" not in result
        assert "<nav>" not in result
        assert "æ­£æåå®¹" in result or "æ®µè½" in result
    
    def test_build_legal_cleaner(self):
        cleaner = build_legal_cleaner()
        
        text = "第一条 法律规定基本制度。\n第二条 本法适用区域及附加条款。"
        result = cleaner.clean(text)
        
        assert "第一条" in result
        assert "第二条" in result
    
    def test_build_enterprise_cleaner(self):
        adapter = build_enterprise_cleaner()
        
        html_text = "<p>HTMLåå®¹æ®µè½</p>"
        html_result = adapter.clean(html_text, doc_type=DocumentType.HTML)
        assert "<p>" not in html_result
        
        md_text = "# æ é¢\n**ç²ä½ææ¬**"
        md_result = adapter.clean(md_text, doc_type=DocumentType.MARKDOWN)
        assert "# " not in md_result
        
        legal_text = "第一条 法律规定具体内容"
        legal_result = adapter.clean(legal_text, doc_type=DocumentType.LEGAL)
        assert "第一条" in legal_result


class TestCleaningLevels:
    
    def test_light_cleaning(self):
        cleaner = SpecialCharCleaner(preserve_chinese=True)
        text = "æµè¯ããææ¬\u200b"
        
        light_result = cleaner.clean(text, level=CleaningLevel.LIGHT)
        standard_result = cleaner.clean(text, level=CleaningLevel.STANDARD)
        
        assert len(light_result) >= len(standard_result)
    
    def test_aggressive_cleaning(self):
        cleaner = SpecialCharCleaner(preserve_chinese=True)
        text = "æµè¯ããææ¬\u200b"
        
        aggressive_result = cleaner.clean(text, level=CleaningLevel.AGGRESSIVE)
        standard_result = cleaner.clean(text, level=CleaningLevel.STANDARD)
        
        assert "\u200b" not in aggressive_result
