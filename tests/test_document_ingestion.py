from soulxpodcast.document import CleaningOptions, clean_tei_for_speech, normalize_for_speech
from soulxpodcast.document.pipeline import _render_document


TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc><titleStmt><title>Example Paper</title></titleStmt></fileDesc>
    <profileDesc><abstract><p>Prior work <ref type="bibr">[1]</ref> matters.</p></abstract></profileDesc>
  </teiHeader>
  <text>
    <body>
      <div><head>1 Introduction</head>
        <p>Transformers are useful <ref type="bibr">[2, 3]</ref>.</p>
        <p>Smith et al. (2021) showed this.</p>
        <figure><head>Figure 1</head><figDesc>An architecture.</figDesc></figure>
        <note place="foot">A footnote that should not be spoken.</note>
      </div>
    </body>
    <back><div type="references"><listBibl><biblStruct>Reference text</biblStruct></listBibl></div></back>
  </text>
</TEI>
"""


def test_cleaner_removes_references_notes_figures_and_inline_citations():
    doc = clean_tei_for_speech(TEI)
    text = _render_document(doc)
    assert "Example Paper" in text
    assert "Prior work matters." in text
    assert "Transformers are useful." in text
    assert "Smith et al. showed this." in text
    assert "Figure 1" not in text
    assert "footnote" not in text
    assert "Reference text" not in text
    assert "[1]" not in text


def test_cleaner_can_keep_citations():
    doc = clean_tei_for_speech(TEI, CleaningOptions(remove_inline_citations=False))
    text = _render_document(doc)
    assert "[1]" in text
    assert "[2, 3]" in text


def test_speech_normalizer_is_conservative():
    assert normalize_for_speech("See Fig. 3 and Eq. 4.") == "See Figure 3 and Equation 4."
