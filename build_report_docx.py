"""Fill the Karunya XAI IA3 report template with the MediXplain report.

    python build_report_docx.py "<path to XAI IA3 Report Template.docx>"

The cover pages, chapter headings, guide bullets and evaluation sheet of the template are
kept as they are. Each <<write the content here>> block is replaced by the chapter's
content, formatted like the reference report: Times New Roman 12 pt, justified, 1.5 line
spacing, 13 pt bold sub-headings, and numbered table / figure captions per chapter.
"""
import json
import re
import sys
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor, Twips
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "results" / "figures"
OUT = ROOT / "MediXplain_IA3_Report.docx"
TITLE = "MediXplain: A Multimodal Explainable-AI Workbench for Clinical Decision Support"
FONT = "Times New Roman"
TEXT_W = 9000  # table width in twips, as in the reference report
BULLET_NUM_ID = 90

M = json.loads((ROOT / "results" / "metrics.json").read_text())
X = json.loads((ROOT / "results" / "report_extras.json").read_text())


# ================================================================ low-level helpers
def set_font(run, size=12, bold=None, italic=None, name=FONT, color="000000"):
    run.font.name = name
    rpr = run._r.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(a), name)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def add_runs(p, text, size=12, bold=False, italic=False, name=FONT):
    """Inline **bold** and *italic* markup."""
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not part:
            continue
        b, i, t = bold, italic, part
        if part.startswith("**"):
            b, t = True, part[2:-2]
        elif part.startswith("*"):
            i, t = True, part[1:-1]
        set_font(p.add_run(t), size=size, bold=b, italic=i, name=name)


def spacing(p, before=0, after=120, line=360):
    pf = p.paragraph_format
    pf.space_before = Twips(before)
    pf.space_after = Twips(after)
    pf.line_spacing = line / 240


class Writer:
    """Inserts formatted blocks immediately before an anchor paragraph."""

    def __init__(self, doc, anchor, chapter):
        self.doc, self.anchor, self.ch = doc, anchor, chapter
        self.tables = self.figures = 0

    def _new_p(self):
        p_el = OxmlElement("w:p")
        self.anchor._p.addprevious(p_el)
        return Paragraph(p_el, self.anchor._parent)

    def sub(self, text):
        p = self._new_p()
        spacing(p, 240, 120)
        p.paragraph_format.keep_with_next = True
        add_runs(p, text, size=13, bold=True)

    def para(self, text):
        p = self._new_p()
        spacing(p)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_runs(p, text)

    def bullets(self, items):
        for t in items:
            p = self._new_p()
            spacing(p, 0, 60)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            ppr = p._p.get_or_add_pPr()
            ppr.insert(0, parse_xml(f'<w:numPr {nsdecls("w")}><w:ilvl w:val="0"/><w:numId w:val="{BULLET_NUM_ID}"/></w:numPr>'))
            add_runs(p, t)

    def caption(self, text, keep_next=False):
        p = self._new_p()
        spacing(p, 60, 240, 276)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = keep_next
        add_runs(p, text, size=11, italic=True)

    def table(self, rows, widths, caption):
        total = sum(widths)
        w = [round(x * TEXT_W / total) for x in widths]
        w[-1] = TEXT_W - sum(w[:-1])
        t = self.doc.add_table(rows=len(rows), cols=len(rows[0]))
        tbl = t._tbl
        tblPr = tbl.tblPr
        for tag in ("w:tblW", "w:jc", "w:tblBorders", "w:tblLayout", "w:tblCellMar"):
            el = tblPr.find(qn(tag))
            if el is not None:
                tblPr.remove(el)
        tblPr.append(parse_xml(f'<w:tblW {nsdecls("w")} w:w="{TEXT_W}" w:type="dxa"/>'))
        tblPr.append(parse_xml(f'<w:jc {nsdecls("w")} w:val="center"/>'))
        b = '<w:{0} w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        tblPr.append(parse_xml(f'<w:tblBorders {nsdecls("w")}>' + "".join(
            b.format(s) for s in ("top", "left", "bottom", "right", "insideH", "insideV")) + "</w:tblBorders>"))
        tblPr.append(parse_xml(f'<w:tblLayout {nsdecls("w")} w:type="fixed"/>'))
        tblPr.append(parse_xml(f'<w:tblCellMar {nsdecls("w")}><w:left w:w="80" w:type="dxa"/>'
                               f'<w:right w:w="80" w:type="dxa"/></w:tblCellMar>'))
        for j, col in enumerate(t.columns):
            col.width = Twips(w[j])
        for i, row in enumerate(rows):
            tr = t.rows[i]._tr
            trPr = tr.get_or_add_trPr()
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>' if i == 0 else f'<w:cantSplit {nsdecls("w")}/>'))
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                cell.width = Twips(w[j])
                if i == 0:
                    cell._tc.get_or_add_tcPr().append(
                        parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="D9E2F3"/>'))
                p = cell.paragraphs[0]
                spacing(p, 40, 40, 240)
                p.paragraph_format.keep_with_next = i < len(rows) - 1
                add_runs(p, str(val), size=10, bold=(i == 0))
        self.anchor._p.addprevious(tbl)
        self.tables += 1
        self.caption(f"Table {self.ch}.{self.tables}: {caption}")

    def figure(self, name, caption, width_cm=15.0):
        p = self._new_p()
        spacing(p, 120, 0, 240)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(FIG / name), width=Cm(width_cm))
        self.figures += 1
        self.caption(f"Figure {self.ch}.{self.figures}: {caption}")

    def code(self, lines):
        for k, line in enumerate(lines):
            p = self._new_p()
            spacing(p, 60 if k == 0 else 0, 60 if k == len(lines) - 1 else 0, 240)
            p.paragraph_format.keep_with_next = k < len(lines) - 1  # keep a code block on one page
            ppr = p._p.get_or_add_pPr()
            ppr.append(parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="F2F2F2"/>'))
            set_font(p.add_run(line if line else " "), size=9, name="Consolas")

    def blank(self):
        self._new_p()


# ================================================================ template surgery
def is_empty(p):
    el = p._p
    return (not p.text.strip() and not el.findall(".//" + qn("w:drawing"))
            and not el.findall(".//" + qn("w:br")) and el.find(qn("w:pPr") + "/" + qn("w:sectPr")) is None)


def is_pagebreak_only(p):
    brs = p._p.findall(".//" + qn("w:br"))
    return (not p.text.strip() and brs and all(b.get(qn("w:type")) == "page" for b in brs)
            and not p._p.findall(".//" + qn("w:drawing")))


def remove(p):
    p._p.getparent().remove(p._p)


def add_bullet_numbering(doc):
    numbering = doc.part.numbering_part.element
    abstract = parse_xml(
        f'<w:abstractNum {nsdecls("w")} w:abstractNumId="{BULLET_NUM_ID}"><w:multiLevelType w:val="singleLevel"/>'
        '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="\u2022"/>'
        '<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr>'
        '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/></w:rPr></w:lvl>'
        "</w:abstractNum>")
    first_num = numbering.find(qn("w:num"))
    if first_num is not None:
        first_num.addprevious(abstract)
    else:
        numbering.append(abstract)
    numbering.append(parse_xml(f'<w:num {nsdecls("w")} w:numId="{BULLET_NUM_ID}">'
                               f'<w:abstractNumId w:val="{BULLET_NUM_ID}"/></w:num>'))


def replace_text_everywhere(doc, old, new):
    for t in doc.element.body.iter(qn("w:t")):
        if old in (t.text or ""):
            t.text = t.text.replace(old, new)


def set_paragraph_text(p, text, highlight=False):
    """Keep the first run's formatting, replace the paragraph's text."""
    runs = p.runs
    first = runs[0]
    for r in runs[1:]:
        r._r.getparent().remove(r._r)
    first.text = text
    if highlight:
        first.font.highlight_color = WD_COLOR_INDEX.YELLOW


def find(doc, text, start=0):
    for i, p in enumerate(doc.paragraphs):
        if i >= start and p.text.strip() == text:
            return i, p
    raise KeyError(text)


# ================================================================ content
f3 = lambda v: f"{v:.3f}"
tm, im, xm, ax = M["tabular"], M["image"], M["text"], X["axioms"]
cf = tm["counterfactual"]


def abstract(w):
    w.para("Machine-learning models now match specialists on many medical tasks, but a prediction that cannot be "
           "explained cannot be safely acted on. This project presents **MediXplain**, a GUI-based explainable-AI "
           "workbench that explains three clinical models, one for each major data type: an XGBoost classifier that "
           "flags malignant breast tumours from 30 biopsy measurements (tabular), a convolutional neural network that "
           "detects pneumonia on chest X-rays (image), and a TF-IDF and logistic-regression model that suggests one of "
           "22 diagnoses from a patient's own description of their symptoms (text). On held-out test data the models "
           f"reach ROC-AUCs of {f3(tm['metrics']['xgboost']['roc_auc'])} and {f3(im['metrics']['roc_auc'])} and a text "
           f"accuracy of {xm['metrics']['accuracy']:.1%}.")
    w.para("The objective is to apply the main families of XAI methods correctly to each data type (Shapley-value "
           "attribution, local surrogates, global importance and dependence plots, counterfactuals, and gradient- and "
           "perturbation-based saliency) and then to go one step further by testing the explanations themselves. "
           "The scope covers model training on three public datasets, fifteen explanation techniques, a quantitative "
           "evaluation of the explanations on held-out data, and an interactive Streamlit application.")
    w.para("Key features include: what-if sliders and a minimal-change counterfactual search for the tumour model; six "
           "attribution maps, a class selector and image upload for X-rays; colour-highlighted words and a contrastive "
           "\"why A and not B\" view for symptom text; plain-English explanation cards generated from the computed "
           "numbers; and a **Trust Lab** that scores every explanation for faithfulness, model sensitivity, agreement "
           "and stability. The Trust Lab found that SmoothGrad and Grad-CAM maps partly survive randomising the CNN's "
           "weights, that LIME's rankings change with its random seed, that the X-ray model treats a fully blurred image "
           "as pneumonia, and that a glass-box logistic regression matches XGBoost on the tabular task (AUC "
           f"{f3(tm['metrics']['logistic_regression']['roc_auc'])} vs {f3(tm['metrics']['xgboost']['roc_auc'])}).")


def chapter1(w):
    w.sub("1.1 Background")
    w.para("Explainable AI (XAI) is the set of methods that make a model's predictions understandable to people. Two "
           "kinds of explanation are usually distinguished. A **local** explanation answers why the model made one "
           "particular prediction; a **global** explanation describes what the model has learned overall. Methods are "
           "also either **model-agnostic**, treating the model as a black box that can only be queried, or "
           "**model-specific**, using its internal structure such as tree splits or gradients. Some models are "
           "interpretable by design (intrinsic interpretability); others need post-hoc explanation.")
    w.para("Healthcare is the domain where explanations matter most. Clinicians are accountable for their decisions, "
           "models can learn shortcuts from artefacts in the data, and regulations such as the EU AI Act require "
           "transparency and human oversight for high-risk systems. Different data types call for different methods: "
           "for tabular data, feature attribution, dependence plots and counterfactuals are natural; for images, "
           "attribution is shown as heatmaps over pixels or regions; for text, it is assigned to words. A complete XAI "
           "study therefore has to handle all three.")
    w.sub("1.2 Problem statement and motivation")
    w.para("A hospital is trialling three AI assistants: biopsy triage, chest X-ray screening and symptom-based triage. "
           "Before deployment, clinicians want to know why each prediction was made and what the models have learned. "
           "A harder problem sits behind this request: explanations are themselves outputs of algorithms, and they can "
           "be wrong. A heatmap that looks convincing but does not depend on the model gives false reassurance. The "
           "problem addressed is therefore twofold: to explain three models built on tabular, image and text data with "
           "suitable methods, and to measure how far each explanation can be trusted.")
    w.sub("1.3 Objectives")
    w.bullets([
        "Train accurate models on tabular, image and text clinical data.",
        "Apply suitable local and global, model-agnostic and model-specific explanation methods to each model.",
        "Generate counterfactual and contrastive explanations that show what would change a decision.",
        "Evaluate the explanations quantitatively: faithfulness, sanity checks, cross-method agreement and stability.",
        "Translate explanations into plain language for non-technical users.",
        "Deliver everything through an interactive GUI application.",
    ])
    w.sub("1.4 Overview of the technologies used")
    w.para("The template asks for an overview of the technologies used (Java, Swing, database connectivity). MediXplain "
           "is a Python machine-learning application with a browser-based GUI, built with the stack in Table 1.1. It "
           "needs no database: trained models are saved as files and results are stored as JSON.")
    w.table([
        ["Layer", "Technology", "Role in the system"],
        ["Language", "Python 3.12", "All modules"],
        ["Tabular model", "XGBoost 3.4, scikit-learn 1.9", "Gradient-boosted trees; glass-box logistic regression"],
        ["Deep learning", "PyTorch 2.13 (CUDA, RTX 3050)", "CNN training and gradient-based attribution"],
        ["Text model", "scikit-learn TF-IDF + LogisticRegression", "22-class symptom classifier"],
        ["XAI libraries", "SHAP 0.54, LIME", "TreeSHAP, LinearSHAP, LIME for tabular, image and text"],
        ["Own implementations", "NumPy, PyTorch, SciPy", "Grad-CAM, IG, SmoothGrad, occlusion, counterfactual search, Trust Lab"],
        ["Visualisation", "Matplotlib, SHAP plots", "All explanation plots and report figures"],
        ["GUI", "Streamlit 1.65", "Six-page interactive web application"],
        ["Data", "scikit-learn, MedMNIST 3.0, Hugging Face", "Three public datasets"],
    ], [1.6, 3.0, 4.4], "Technologies used in MediXplain")


def chapter2(w):
    w.sub("2.1 Interpretability and a taxonomy of methods")
    w.para("Doshi-Velez and Kim (2017) define interpretability as the ability to explain or present a model in "
           "understandable terms to a human, and argue that it must be evaluated, not assumed. Lipton (2018) warns that "
           "\"interpretability\" covers several different goals: trust, causality, transferability and fairness. "
           "Methods are commonly organised along three axes, summarised in Table 2.1, which is also how MediXplain "
           "organises its methods.")
    w.table([
        ["", "Local (one prediction)", "Global (whole model)"],
        ["**Model-agnostic**", "LIME (tabular, image, text), occlusion, counterfactuals", "Permutation importance, PDP / ICE"],
        ["**Model-specific**", "TreeSHAP, Grad-CAM, Integrated Gradients, SmoothGrad, saliency", "Mean |SHAP|, SHAP beeswarm and dependence"],
        ["**Intrinsic**", "Logistic-regression contributions wⱼxⱼ", "Coefficients"],
    ], [1.6, 4.0, 3.4], "Taxonomy of the explanation methods used")
    w.sub("2.2 Feature attribution: SHAP and LIME")
    w.para("Shapley values (Shapley, 1953) share a game's payout fairly among players. Lundberg and Lee (2017) used them "
           "to attribute a prediction to features, proving that SHAP is the only additive attribution with local "
           "accuracy, missingness and consistency. TreeSHAP (Lundberg et al., 2020) computes exact Shapley values for "
           "tree ensembles in polynomial time. LIME (Ribeiro, Singh and Guestrin, 2016) instead samples perturbed inputs "
           "around the instance and fits a weighted sparse linear model. LIME works with any model but is random and "
           "only locally valid: Alvarez-Melis and Jaakkola (2018) showed that its explanations can change for nearly "
           "identical inputs, and Slack et al. (2020) showed that both LIME and SHAP can be fooled by an adversarial "
           "model. MediXplain therefore measures LIME's stability directly.")
    w.sub("2.3 Global explanations")
    w.para("Permutation importance (Breiman, 2001; Fisher, Rudin and Dominici, 2019) measures the drop in performance "
           "when one feature is shuffled. Partial dependence plots (Friedman, 2001) show the average model response as "
           "one feature varies; individual conditional expectation curves (Goldstein et al., 2015) show the same for "
           "each instance and reveal effects that averaging hides. Both assume features can be varied independently, "
           "which is unrealistic for correlated features such as tumour radius, perimeter and area (Apley and Zhu, 2020).")
    w.sub("2.4 Counterfactual explanations")
    w.para("Wachter, Mittelstadt and Russell (2017) proposed explaining a decision by the smallest change to the input "
           "that would alter it, a form that people find natural and that needs no access to model internals. DiCE "
           "(Mothilal, Sharma and Tan, 2020) generates several diverse counterfactuals, and Ustun, Spangher and Liu "
           "(2019) distinguish actionable recourse from mere explanation. MediXplain's counterfactuals on biopsy "
           "measurements are explanatory, since a patient cannot change their cell nuclei.")
    w.sub("2.5 Explaining convolutional networks")
    w.para("Simonyan, Vedaldi and Zisserman (2014) introduced gradient saliency maps; Zeiler and Fergus (2014) used "
           "occlusion. SmoothGrad (Smilkov et al., 2017) averages gradients over noisy copies to reduce visual noise. "
           "Integrated Gradients (Sundararajan, Taly and Yan, 2017) integrates gradients along a path from a baseline "
           "and satisfies completeness. Grad-CAM (Selvaraju et al., 2017) weights the last convolutional feature maps by "
           "their average gradient and produces coarse class-specific heatmaps. In medical imaging, CheXNet (Rajpurkar "
           "et al., 2017) used class activation maps for pneumonia, while Zech et al. (2018) and DeGrave, Janizek and Lee "
           "(2021) showed that chest X-ray models learn shortcuts such as hospital-specific markers, which is exactly "
           "what explanations should expose.")
    w.sub("2.6 Evaluating explanations")
    w.para("Adebayo et al. (2018) proposed sanity checks: a saliency map should change when the model's weights are "
           "randomised, and several popular methods fail. Samek et al. (2017) and Petsiuk, Das and Saenko (2018) "
           "evaluate faithfulness by deleting the most important pixels and tracking the prediction. Hooker et al. "
           "(2019) point out that deleted pixels create out-of-distribution inputs, an effect MediXplain ran into "
           "directly (Section 6.7). For text, ERASER (DeYoung et al., 2020) defines comprehensiveness as the drop in "
           "probability when the rationale words are removed. Rudin (2019) argues that for high-stakes decisions an "
           "interpretable model should be preferred whenever it performs as well as a black box.")
    w.sub("2.7 Comparison with existing tools")
    w.para("Several libraries implement parts of this pipeline. Table 2.2 compares them with MediXplain. The libraries "
           "provide the explanation algorithms, while MediXplain combines all three data types, evaluates the "
           "explanations and presents them in one GUI.")
    w.table([
        ["Tool", "Tabular", "Image", "Text", "Counterfactuals", "Evaluates explanations", "Interactive GUI"],
        ["SHAP library", "Yes", "Yes", "Yes", "—", "Partial", "—"],
        ["LIME library", "Yes", "Yes", "Yes", "—", "—", "—"],
        ["Captum (PyTorch)", "Yes", "Yes", "Yes", "—", "Partial", "Partial"],
        ["InterpretML", "Yes", "—", "—", "—", "—", "Yes"],
        ["explainerdashboard", "Yes", "—", "—", "Partial (what-if)", "—", "Yes"],
        ["**MediXplain (this project)**", "Yes", "Yes", "Yes", "Yes", "Yes (Trust Lab)", "Yes"],
    ], [2.4, 1.0, 1.0, 1.0, 1.5, 1.6, 1.3], "XAI tools compared (Partial = limited support)")


def chapter3(w):
    w.sub("3.1 Architecture and high-level design")
    w.para("MediXplain is organised in four layers (Figure 3.1). The data layer holds the three datasets. Each modality "
           "module pairs a model with the explainers that suit it. The Trust Lab evaluates the explanations, and the "
           "presentation layer turns them into plain-English cards and interactive pages. An offline pipeline "
           "(train_all.py) trains the models, runs every Trust Lab test and saves metrics and figures; the GUI loads the "
           "saved models and computes explanations on demand.")
    w.figure("arch_overview.png", "Layered architecture of MediXplain", 15.0)
    w.sub("3.2 Modules")
    w.table([
        ["Module", "File", "Responsibility"],
        ["Tabular", "xai/tabular.py", "Train XGBoost and glass-box LR; SHAP, LIME, PDP/ICE, permutation importance, counterfactual search"],
        ["Image", "xai/image.py", "Train the CNN; Grad-CAM, IG, SmoothGrad, saliency, occlusion, LIME; deletion curves, weight randomisation"],
        ["Text", "xai/text.py", "Train TF-IDF + LR; LIME, SHAP, word occlusion, linear and contrastive contributions, highlighting"],
        ["Trust Lab", "xai/trust.py", "Faithfulness, sanity, agreement and stability tests for all three modalities"],
        ["Narrative", "xai/narrative.py", "Plain-English explanation cards filled from computed numbers"],
        ["Pipeline", "train_all.py", "Train all models, run all tests, save results/metrics.json and figures"],
        ["GUI", "app.py", "Six-page Streamlit application with cached models and explanations"],
    ], [1.3, 1.8, 5.9], "Modules and responsibilities")
    w.sub("3.3 Explanation flow")
    w.para("Every explanation request follows the same path (Figure 3.2). The user chooses a case, uploads an image or "
           "types text; the model predicts; the explainers compute attributions, cached so that repeated views are "
           "instant; trust checks such as the deletion curve and the sanity check run on the same input; and the result "
           "is shown as plots and a plain-English card.")
    w.figure("arch_flow.png", "Flow of one explanation request", 15.0)
    w.sub("3.4 Explanation methods by data type")
    w.table([
        ["Method", "Data", "Scope", "Type", "Output"],
        ["TreeSHAP", "Tabular", "Local + global", "Model-specific", "Waterfall, beeswarm, dependence"],
        ["LIME", "Tabular, image, text", "Local", "Model-agnostic", "Rule weights, superpixels, word weights"],
        ["Permutation importance", "Tabular", "Global", "Model-agnostic", "Drop in test AUC"],
        ["PDP / ICE", "Tabular", "Global", "Model-agnostic", "Response curves"],
        ["Counterfactual search", "Tabular", "Local", "Model-agnostic", "Minimal feature changes"],
        ["Glass-box coefficients", "Tabular", "Local + global", "Intrinsic", "wⱼxⱼ contributions"],
        ["Grad-CAM", "Image", "Local", "Model-specific", "Coarse class heatmap"],
        ["Integrated Gradients", "Image", "Local", "Model-specific", "Signed pixel attribution"],
        ["SmoothGrad / saliency", "Image", "Local", "Model-specific", "Pixel gradient magnitude"],
        ["Occlusion", "Image, text", "Local", "Model-agnostic", "Probability drop per patch or word"],
        ["LinearSHAP", "Text", "Local", "Model-specific", "Word attributions"],
        ["Contrastive", "Text", "Local", "Intrinsic", "Why class A rather than B"],
    ], [2.2, 1.7, 1.4, 1.6, 2.6], "Explanation methods applied to each data type")
    w.sub("3.5 User interface design")
    w.para("The template asks for an interface designed with Swing. MediXplain's GUI is a browser application built "
           "with Streamlit, with a sidebar to switch between six pages (Table 3.5). Each case page has the same layout: "
           "the input and prediction on the left with a plain-English card, and tabs of explanations on the right.")
    w.table([
        ["Page", "Purpose", "Main interactions"],
        ["Overview", "Problem, three models, what is new", "Model metrics at a glance"],
        ["1 · Tumour (tabular)", "Explain malignancy predictions", "Patient selector, what-if sliders, LIME seed, SHAP / LIME / counterfactual / global tabs"],
        ["2 · Chest X-ray (image)", "Explain pneumonia predictions", "Test image or upload, class to explain, method selector, faithfulness and sanity tabs"],
        ["3 · Symptoms (text)", "Explain diagnosis suggestions", "Free text, method selector, word highlighting, contrastive class, global words"],
        ["4 · Trust Lab", "Test the explanations", "Faithfulness, sanity, agreement and stability results; live stability test"],
        ["Concepts & methods", "Theory reference", "Taxonomy, formulas, limitations and ethics"],
    ], [1.9, 2.6, 4.5], "Pages of the GUI")


def chapter4(w):
    w.sub("4.1 Data preparation")
    w.para("**Tabular.** The Wisconsin Diagnostic Breast Cancer data (Street, Wolberg and Mangasarian, 1993) has 569 "
           "biopsies with 30 measurements of cell nuclei (mean, standard error and worst value of ten properties). It "
           "was split 75/25 with stratification; malignant is the positive class. **Image.** PneumoniaMNIST (Yang et "
           "al., 2023) is derived from 5,856 paediatric chest X-rays (Kermany et al., 2018), centre-cropped and resized "
           "to 64×64 grayscale. The official split has 4,708 training, 524 validation and 624 test images; 74% of the "
           "training images show pneumonia. **Text.** The symptom_to_diagnosis dataset contains 1,065 first-person "
           "symptom descriptions labelled with 22 diagnoses, with an official 853/212 split.")
    w.sub("4.2 Model training")
    w.table([
        ["Model", "Configuration", "Training"],
        ["XGBoost", "300 trees, depth 3, learning rate 0.05, subsample 0.9, column sample 0.8", "Log-loss, fixed seed 42"],
        ["Glass-box LR", "StandardScaler + L2 logistic regression, C = 0.5", "Same split as XGBoost"],
        ["CNN", "4 × (conv 3×3, batch norm, ReLU), max-pooling, global average pooling, dropout 0.3, linear", "AdamW, one-cycle LR (max 3e-3), class-weighted loss, 15 epochs, random shifts and brightness jitter; best validation AUC kept"],
        ["Text classifier", "TF-IDF 1–2 grams, min_df 2, sublinear TF, English stop words; multinomial LR, C = 10", "Official train split"],
    ], [1.5, 4.2, 3.3], "Model configurations")
    w.sub("4.3 Tabular explainers")
    w.para("TreeSHAP explains XGBoost in log-odds of malignancy, so each patient's SHAP values add up exactly to the "
           "model's output minus the average output. LIME uses quartile-discretised features, which turns its weights "
           "into readable rules such as \"area_error > 45.39\", with 2,000 samples per explanation. Permutation "
           "importance uses 20 repeats and the drop in test ROC-AUC. PDP and ICE curves use 60 sampled test patients. "
           "The glass-box reference gives each feature the contribution wⱼxⱼ on standardised features.")
    w.sub("4.4 Counterfactual search")
    w.para("A greedy search looks for a minimal change that flips the prediction. At each step every feature may move "
           "0.25, 0.5 or 1 standard deviation toward the median of the opposite class, never leaving the range seen in "
           "training. The move with the largest probability gain per unit of standardised distance is kept. Because "
           "tree ensembles are piecewise constant, some moves change nothing; when no move helps, the search takes a "
           "step on the feature with the largest |SHAP| value to cross the plateau. The search stops when the class "
           "flips or after 80 steps, and reports every changed feature with its old and new value (Appendix C).")
    w.sub("4.5 Image explainers")
    w.para("All image methods were written directly in PyTorch so each could be explained line by line. **Grad-CAM** "
           "hooks the last convolutional block (128 maps of 8×8), averages the gradients of the class score over each "
           "map, takes ReLU of the weighted sum and upsamples it to 64×64 (Appendix B). **Vanilla gradients** take the "
           "absolute input gradient; **SmoothGrad** averages it over 30 noisy copies (σ = 0.1). **Integrated "
           "Gradients** uses a black baseline and a 64-step midpoint Riemann sum. **Occlusion** slides an 8×8 patch "
           "filled with the image mean with stride 4 and records the drop in probability. **LIME** perturbs about 40 "
           "SLIC superpixels with 800 samples.")
    w.sub("4.6 Text explainers")
    w.para("Because the text model is linear in TF-IDF features, its exact contribution for each n-gram is the TF-IDF "
           "value times the class coefficient; the contrastive view uses the difference of two class coefficients. "
           "LinearSHAP uses an independent masker with the whole training set as background. LIME text uses 1,500 "
           "samples. Occlusion deletes each distinct word and records the drop in probability. Bigram scores are split "
           "between their two words so that all methods can be compared word by word and shown as highlighted text.")
    w.sub("4.7 Trust Lab")
    w.para("**Faithfulness.** Tabular: the top-5 supporting features are reset to the training mean and the drop in "
           "the predicted class's probability is measured. Image: pixels are removed in 20 steps, most important "
           "first, by replacing them with a blurred copy of the image (σ = 8), and the area under the probability "
           "curve is recorded (lower is better). Text: the top-3 words are deleted (ERASER comprehensiveness). Every "
           "test includes a random baseline. **Sanity check.** All convolutional and linear layers are re-initialised, "
           "and the Spearman correlation between the original and randomised maps is computed. **Agreement.** Top-k "
           "overlap and Spearman rank correlation between methods. **Stability.** Five LIME runs with different seeds, "
           "and five copies of the input with 1% Gaussian noise.")
    w.sub("4.8 Plain-language cards and GUI")
    w.para("narrative.py builds a short card for each prediction from the computed numbers only: the prediction and "
           "confidence, the strongest evidence for and against it, the counterfactual (tabular) or the region the model "
           "looked at (image), and a reminder that a clinician decides (examples in Appendix D). The Streamlit app "
           "caches models with st.cache_resource and attribution maps with st.cache_data, so switching tabs is instant. "
           "Uploaded X-rays are converted to grayscale, centre-cropped and resized to 64×64. Pages can be opened "
           "directly with a ?page=N link, which was also used to capture the screenshots in this report.")


def chapter5(w):
    w.sub("5.1 Testing approach")
    w.para("Testing was done at four levels: validation of the models on held-out test sets; property tests that "
           "check the explainers satisfy their mathematical guarantees; evaluation of the explanations with the Trust "
           "Lab; and functional testing of every GUI page in a browser. All randomness uses a fixed seed (42); GPU "
           "training is not fully deterministic, so CNN scores can vary slightly between runs.")
    w.sub("5.2 Model validation")
    w.table([
        ["Model", "Test set", "Accuracy", "F1", "ROC-AUC"],
        ["XGBoost (tabular)", f"{tm['n_test']} biopsies", f3(tm['metrics']['xgboost']['accuracy']),
         f3(tm['metrics']['xgboost']['f1']), f3(tm['metrics']['xgboost']['roc_auc'])],
        ["Logistic regression (glass box)", f"{tm['n_test']} biopsies", f3(tm['metrics']['logistic_regression']['accuracy']),
         f3(tm['metrics']['logistic_regression']['f1']), f3(tm['metrics']['logistic_regression']['roc_auc'])],
        ["CNN (X-ray)", "624 X-rays", f3(im['metrics']['accuracy']), f3(im['metrics']['f1']), f3(im['metrics']['roc_auc'])],
        ["TF-IDF + LR (text)", f"{xm['n_test']} descriptions", f3(xm['metrics']['accuracy']),
         f"{f3(xm['metrics']['macro_f1'])} (macro)", "—"],
    ], [2.8, 1.8, 1.4, 1.6, 1.4], "Model performance on held-out test data")
    w.sub("5.3 Test cases for the explainers")
    w.table([
        ["Test case", "Expected", "Result", "Status"],
        ["SHAP local accuracy", "Σφ + E[f] = model log-odds", f"Max error {ax['shap_additivity_max_abs_error']:.1e} over {ax['shap_n']} patients", "Pass"],
        ["IG completeness", "ΣIG = f(x) − f(baseline)", f"Mean relative error {ax['ig_completeness_mean_rel_error']:.2%} over {ax['ig_n']} X-rays", "Pass"],
        ["TreeSHAP determinism", "Same input gives the same explanation", "Spearman 1.000 across runs and under 1% noise", "Pass"],
        ["Counterfactual validity", "Class flips, values stay in training range", f"{cf['flip_rate']:.0%} of 40 test patients flipped; {cf['mean_features_changed']:.1f} features changed on average", "Pass"],
        ["Tabular faithfulness", "Beats random features", f"SHAP {f3(tm['faithfulness_top5_drop']['SHAP'])} vs random {f3(tm['faithfulness_top5_drop']['Random'])}", "Pass"],
        ["Image deletion", "Every method below random AUC", f"{min(v for k, v in im['deletion_auc'].items() if k != 'Random'):.3f}–{max(v for k, v in im['deletion_auc'].items() if k != 'Random'):.3f} vs random {f3(im['deletion_auc']['Random'])}", "Pass"],
        ["Text comprehensiveness", "Beats random words", f"≥ {min(v for k, v in xm['comprehensiveness_top3'].items() if k != 'Random'):.3f} vs random {f3(xm['comprehensiveness_top3']['Random'])}", "Pass"],
        ["Sanity check", "Maps change with random weights", "Occlusion, saliency, IG pass; Grad-CAM 0.34, SmoothGrad 0.45", "Partial"],
        ["LIME seed stability", "Ranking independent of seed", f"Spearman {tm['stability']['LIME seed-to-seed spearman']:.2f} (tabular), {xm['lime_seed_stability']:.2f} (text)", "Partial"],
        ["Image upload", "Any JPG/PNG becomes a valid 64×64 input", "Prediction and all maps computed", "Pass"],
    ], [2.0, 2.4, 3.5, 1.1], "Test cases for the explanation methods")
    w.para("The two partial results are findings rather than defects: they describe limitations of the methods "
           "themselves, which the GUI makes visible so that users do not over-trust them (Section 6.5).")
    w.sub("5.4 Functional testing of the GUI")
    w.para("Every page was opened in a browser and every tab exercised. Two defects were found and fixed: on the text "
           "page the method selector was drawn below the results instead of above them, and the tumour card read the "
           "SHAP baseline before it was final, showing a 37% instead of a 40% starting risk.")
    w.table([
        ["Page", "Checks performed", "Result"],
        ["Overview", "Metrics load from results/metrics.json", "Pass"],
        ["Tumour", "Patient selector, what-if sliders, SHAP, LIME (seed change), counterfactual table, global plots", "Pass"],
        ["Chest X-ray", "Test images and upload, class switch, six maps, deletion curves, sanity check", "Pass"],
        ["Symptoms", "Free text, four highlighting methods, contrastive class, global words", "Pass (after layout fix)"],
        ["Trust Lab", "All charts and tables, live stability test", "Pass"],
        ["Concepts & methods", "Taxonomy table and formulas render", "Pass"],
    ], [1.6, 5.6, 1.8], "GUI functional tests")
    w.sub("5.5 Validation against the requirements")
    w.table([
        ["IA3 rubric criterion", "Evidence in this project"],
        ["Concept understanding", "Taxonomy and method theory (Chapter 2, Concepts page); axiom tests (Table 5.3)"],
        ["Applying to data", "Tabular, image and text models, each with several suitable methods (Table 3.4)"],
        ["Visualization & interpretation", "Over 20 interactive plots, highlighted text, plain-English cards (Chapter 6)"],
        ["Presentation", "Six-page GUI with a guided overview and a demo flow (Appendix A)"],
        ["Innovation", "Trust Lab, counterfactual search, glass-box comparison, contrastive text, shortcut discovery"],
        ["Documentation", "This report, README and REPORT.md in the project folder"],
    ], [2.6, 6.4], "Requirements and how they were met")


def chapter6(w):
    w.sub("6.1 Presentation of the final system")
    w.para("The figures below show the final application. Figure 6.1 is the overview page; Figures 6.2 to 6.7 show the "
           "three case pages and the Trust Lab.")
    for name, cap in [
        ("gui_overview.png", "Overview page: the three models and their test performance"),
        ("gui_tumour_shap.png", "Tumour page: prediction, plain-English card and SHAP waterfall"),
        ("gui_tumour_counterfactual.png", "Tumour page: counterfactual that flips the prediction to benign"),
        ("gui_xray_maps.png", "Chest X-ray page: six attribution maps for a pneumonia case"),
        ("gui_xray_sanity.png", "Chest X-ray page: sanity check against a CNN with random weights"),
        ("gui_text.png", "Symptoms page: SHAP word highlighting and the contrastive view"),
        ("gui_trust_lab.png", "Trust Lab page: faithfulness and sanity-check results"),
    ]:
        w.figure(name, cap, 15.5)
    w.sub("6.2 Tabular explanations")
    w.para("All three global views rank the *worst* (largest-nucleus) measurements highest: concave points, perimeter, "
           "area and radius (Figure 6.8). Permutation importance is more concentrated, because correlated features share "
           "credit when one is shuffled. The beeswarm (Figure 6.9) shows the direction of each effect: high concave-point "
           "and area values push toward malignant.")
    w.figure("tab_global_importance.png", "Global importance: mean |SHAP|, permutation importance and glass-box coefficients", 15.5)
    w.figure("tab_shap_beeswarm.png", "SHAP beeswarm over the test patients", 12.5)
    w.para("In the PDP/ICE plot for worst concave points (Figure 6.10), the average curve rises only from 0.32 to 0.40, "
           "but individual borderline patients jump sharply near 0.135 and 0.16–0.175. Averaging hides these strong "
           "individual effects, which is why ICE is shown alongside PDP.")
    w.figure("tab_pdp_ice.png", "Partial dependence (red) and ICE curves (grey) for worst concave points", 13.0)
    w.para("For test patient #0 (P(malignant) = 99.8%), mean and worst concave points add +1.20 and +1.04 log-odds and "
           "worst perimeter +0.88 (Figure 6.11). LIME selects the same features as readable rules. The counterfactual "
           f"search flips this patient to benign (P = {cf['example']['end_prob']:.1%}) by lowering six measurements, "
           "for example worst perimeter from 151.7 to 100.2 and worst area from 1332 to 597. Across 40 test patients the "
           f"search flipped {cf['flip_rate']:.0%} of predictions, changing {cf['mean_features_changed']:.1f} features "
           f"with a total move of {cf['mean_l1_std_distance']:.1f} standard deviations on average.")
    w.figure("tab_shap_waterfall.png", "SHAP waterfall for test patient #0", 13.0)
    w.sub("6.3 Image explanations")
    w.para("On the pneumonia case (Figure 6.12), Grad-CAM and occlusion focus on the central lung fields. Gradient maps "
           "(vanilla, IG) are noisy and pixel-level, SmoothGrad spreads over the whole lung area, and LIME's superpixels "
           "agree with occlusion on the central-left region. On the example X-ray the methods form two families: "
           "region-based maps (Grad-CAM, occlusion, LIME, SmoothGrad; ρ ≈ 0.4–0.6) and pixel-gradient maps (vanilla, "
           "IG; ρ ≈ 0.45 with each other and below 0.1 with Grad-CAM).")
    w.figure("img_maps_pneumonia.png", "Six attribution methods on a pneumonia X-ray", 15.5)
    w.figure("img_maps_normal.png", "The same methods on a normal X-ray", 15.5)
    w.sub("6.4 Text explanations")
    w.para("For \"I'm sweating a lot and can't catch my breath… phlegm… my heart is racing… coughing up brown stuff\", "
           "the model predicts pneumonia (96%). SHAP, LIME and occlusion all highlight heart, racing, phlegm, sweating "
           "and brown (Figures 6.14 and 6.15). LIME gives throat a negative weight, because throat words usually point "
           "to a cold or allergy, and the contrastive view against GERD shows throat as the only word favouring the "
           "alternative. Across 40 test descriptions, SHAP and the exact linear contributions share "
           f"{xm['agreement_top3']['SHAP vs Linear contribution']:.0%} of their top-3 words.")
    w.figure("txt_shap.png", "SHAP word attributions for the pneumonia prediction", 10.5)
    w.figure("txt_lime.png", "LIME word weights for the same prediction", 10.5)
    w.sub("6.5 Trust Lab results")
    fd, da, cp = tm["faithfulness_top5_drop"], im["deletion_auc"], xm["comprehensiveness_top3"]
    w.table([
        ["Data", "Test", "Method scores", "Random"],
        ["Tabular", "Probability drop, top-5 features reset (↑)", f"SHAP {f3(fd['SHAP'])} · LIME {f3(fd['LIME'])} · glass-box {f3(fd['Glass-box coef'])}", f3(fd["Random"])],
        ["Image", "Deletion-curve AUC, 30 X-rays (↓)", " · ".join(f"{k} {f3(v)}" for k, v in sorted(da.items(), key=lambda kv: kv[1]) if k != "Random"), f3(da["Random"])],
        ["Text", "ERASER comprehensiveness, top-3 words (↑)", " · ".join(f"{k} {f3(v)}" for k, v in sorted(cp.items(), key=lambda kv: -kv[1]) if k != "Random"), f3(cp["Random"])],
    ], [1.0, 2.6, 4.5, 0.9], "Faithfulness of every method against a random baseline")
    w.para("Every method beats the random baseline on every data type. Occlusion is the most faithful image method, "
           "because it measures exactly what the deletion test measures. Several deletion curves climb back to "
           "P(pneumonia) ≈ 1 when almost every pixel has been replaced by the blurred image (Figure 6.16): a fully "
           "blurred X-ray is classified as pneumonia. The CNN has partly learned that a hazy, low-contrast image means "
           "pneumonia, a shortcut a radiologist should be told about.")
    w.figure("img_deletion_curves.png", "Deletion curves on one X-ray (a faster drop means a more faithful map)", 12.5)
    sp = im["sanity_spearman"]
    verdict = lambda v: "Passes" if v < 0.15 else "Mostly passes" if v < 0.25 else "Partly reflects image structure" if v < 0.4 else "Similar even with random weights"
    w.table([["Method", "Similarity to random-weight map (↓)", "Verdict"]] +
            [[k, f3(v), verdict(v)] for k, v in sorted(sp.items(), key=lambda kv: kv[1])],
            [2.6, 3.0, 3.4], "Model-parameter randomisation sanity check (Adebayo et al., 2018)")
    w.figure("img_sanity.png", "The same methods computed on a CNN with random weights", 15.5)
    w.para("SmoothGrad produces the most visually appealing maps, yet nearly half of its structure survives destroying "
           "the model; visual appeal is not evidence of faithfulness. Agreement and stability results are in Table 6.3.")
    ag, st = tm["agreement"], tm["stability"]
    w.table([
        ["Test", "Result"],
        ["SHAP vs LIME (40 patients)", f"Top-5 overlap {ag['SHAP vs LIME']['top5_overlap']:.2f}, Spearman {ag['SHAP vs LIME']['spearman']:.2f}"],
        ["SHAP vs glass-box coefficients", f"Top-5 overlap {ag['SHAP vs glass-box']['top5_overlap']:.2f}, Spearman {ag['SHAP vs glass-box']['spearman']:.2f}"],
        ["Text: SHAP vs LIME / occlusion (top-3)", f"{xm['agreement_top3']['SHAP vs LIME']:.2f} / {xm['agreement_top3']['SHAP vs Occlusion']:.2f}"],
        ["TreeSHAP: new seed / 1% input noise", f"Spearman {st['SHAP seed-to-seed spearman']:.3f} / {st['SHAP under 1% input noise']:.3f}"],
        ["LIME tabular: new seed / 1% input noise", f"Spearman {st['LIME seed-to-seed spearman']:.3f} / {st['LIME under 1% input noise']:.3f}"],
        ["LIME text: new seed", f"Spearman {xm['lime_seed_stability']:.3f}"],
    ], [4.0, 5.0], "Agreement between methods and stability of explanations")
    w.sub("6.6 Evaluation against the objectives")
    w.para("Every objective in Section 1.3 was met. Three accurate models were trained on tabular, image and text "
           "data; each is explained with local and global, agnostic and specific methods; counterfactual and "
           "contrastive views show what would change a decision; the Trust Lab measures faithfulness, sanity, "
           "agreement and stability; plain-English cards summarise each prediction; and the whole system runs as an "
           "interactive GUI. The evaluation also produced an argument for interpretable models: the glass-box logistic "
           f"regression matches XGBoost (AUC {f3(tm['metrics']['logistic_regression']['roc_auc'])} vs "
           f"{f3(tm['metrics']['xgboost']['roc_auc'])}), as Rudin (2019) recommends, although the two models rely on "
           f"different features (top-5 overlap only {ag['SHAP vs glass-box']['top5_overlap']:.2f}).")
    w.sub("6.7 Challenges faced")
    w.bullets([
        "**Misleading deletion metric.** Filling removed pixels with flat grey made random maps look most faithful, "
        "because scattered grey pixels act as noise for the CNN; a blurred fill (σ = 8) fixed this.",
        "**Counterfactual plateaus.** Tree ensembles are piecewise constant, so the first greedy search stalled; larger "
        "steps and SHAP-guided plateau moves now flip every test case.",
        "**Saturated probabilities.** The CNN is often more than 99.9% confident, so occlusion changes are tiny; maps "
        "are normalised for display and compared with deletion curves.",
        "**Class imbalance.** 74% of training X-rays show pneumonia; a class-weighted loss and AUC-based checkpointing "
        "prevent the model from always predicting pneumonia.",
        "**LIME randomness.** Different seeds gave different rankings; the GUI exposes the seed and the Trust Lab "
        "measures the effect.",
        f"**Validation-to-test gap.** The CNN reached a validation AUC of {im['metrics']['val_auc']:.3f} but "
        f"{im['metrics']['roc_auc']:.3f} on the test set, which comes from a separate split of the source data.",
    ])


def conclusion(w):
    w.sub("Project summary")
    w.para("MediXplain shows that one GUI application can explain clinical models built on tabular, image and text "
           "data with fifteen techniques, and can test whether those explanations deserve trust. Its main lesson is "
           "that explanations must be evaluated like models: some widely used saliency maps partly ignore the model, "
           "LIME depends on its seed, and even the evaluation metric needed a sanity check before it could be used.")
    w.sub("Key achievements and limitations")
    w.bullets([
        "Achieved: accurate models on three data types; SHAP, LIME, PDP/ICE, permutation importance, counterfactuals, "
        "Grad-CAM, IG, SmoothGrad, saliency, occlusion and contrastive text explanations; a Trust Lab; plain-English "
        "cards; and a six-page GUI.",
        "Achieved: discovered a shortcut in the X-ray model and showed that an interpretable model is sufficient for "
        "the tabular task.",
        "Limitation: X-rays are 64×64 and the symptom texts are synthetic phrasing; no model is clinically validated.",
        "Limitation: the counterfactual search is greedy, so its results are not guaranteed to be minimal.",
        "Limitation: explanation tests use 20–40 cases per modality and a single training run.",
    ])
    w.sub("Recommended enhancements")
    w.bullets([
        "Use full-resolution X-rays with a pretrained DenseNet, as in CheXNet, and test for hospital-marker shortcuts.",
        "Generate diverse, plausibility-constrained counterfactuals with DiCE.",
        "Add retraining-based evaluation (ROAR) to avoid the distribution shift of pixel deletion.",
        "Replace the text model with a clinical transformer and explain it with Integrated Gradients and attention.",
        "Store every prediction and explanation in MongoDB to give an audit trail for clinical governance.",
        "Run a user study with clinicians to measure whether the explanations improve their decisions.",
    ])


def references(w):
    w.bullets([
        "Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M. and Kim, B. (2018). Sanity checks for saliency maps. Advances in Neural Information Processing Systems 31.",
        "Alvarez-Melis, D. and Jaakkola, T. S. (2018). On the robustness of interpretability methods. arXiv:1806.08049.",
        "Apley, D. W. and Zhu, J. (2020). Visualizing the effects of predictor variables in black box supervised learning models. Journal of the Royal Statistical Society Series B, 82(4), 1059–1086.",
        "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.",
        "DeGrave, A. J., Janizek, J. D. and Lee, S.-I. (2021). AI for radiographic COVID-19 detection selects shortcuts over signal. Nature Machine Intelligence, 3, 610–619.",
        "DeYoung, J. et al. (2020). ERASER: A benchmark to evaluate rationalized NLP models. Proceedings of ACL 2020.",
        "Doshi-Velez, F. and Kim, B. (2017). Towards a rigorous science of interpretable machine learning. arXiv:1702.08608.",
        "European Parliament and Council (2024). Regulation (EU) 2024/1689 — Artificial Intelligence Act.",
        "Fisher, A., Rudin, C. and Dominici, F. (2019). All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously. Journal of Machine Learning Research, 20(177).",
        "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189–1232.",
        "Goldstein, A., Kapelner, A., Bleich, J. and Pitkin, E. (2015). Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation. Journal of Computational and Graphical Statistics, 24(1), 44–65.",
        "Gretel.ai (n.d.). symptom_to_diagnosis [dataset]. Hugging Face. https://huggingface.co/datasets/gretelai/symptom_to_diagnosis",
        "Hooker, S., Erhan, D., Kindermans, P.-J. and Kim, B. (2019). A benchmark for interpretability methods in deep neural networks. Advances in Neural Information Processing Systems 32.",
        "Kermany, D. S. et al. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. Cell, 172(5), 1122–1131.",
        "Lipton, Z. C. (2018). The mythos of model interpretability. Communications of the ACM, 61(10), 36–43.",
        "Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems 30.",
        "Lundberg, S. M. et al. (2020). From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence, 2, 56–67.",
        "Mothilal, R. K., Sharma, A. and Tan, C. (2020). Explaining machine learning classifiers through diverse counterfactual explanations. Proceedings of FAT* 2020.",
        "Petsiuk, V., Das, A. and Saenko, K. (2018). RISE: Randomized input sampling for explanation of black-box models. Proceedings of BMVC 2018.",
        "Rajpurkar, P. et al. (2017). CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning. arXiv:1711.05225.",
        "Ribeiro, M. T., Singh, S. and Guestrin, C. (2016). \"Why should I trust you?\": Explaining the predictions of any classifier. Proceedings of KDD 2016.",
        "Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence, 1, 206–215.",
        "Samek, W., Binder, A., Montavon, G., Lapuschkin, S. and Müller, K.-R. (2017). Evaluating the visualization of what a deep neural network has learned. IEEE Transactions on Neural Networks and Learning Systems, 28(11), 2660–2673.",
        "Selvaraju, R. R. et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. Proceedings of ICCV 2017.",
        "Shapley, L. S. (1953). A value for n-person games. Contributions to the Theory of Games II, 307–317.",
        "Simonyan, K., Vedaldi, A. and Zisserman, A. (2014). Deep inside convolutional networks: Visualising image classification models and saliency maps. ICLR Workshop 2014.",
        "Slack, D., Hilgard, S., Jia, E., Singh, S. and Lakkaraju, H. (2020). Fooling LIME and SHAP: Adversarial attacks on post hoc explanation methods. Proceedings of AIES 2020.",
        "Smilkov, D., Thorat, N., Kim, B., Viégas, F. and Wattenberg, M. (2017). SmoothGrad: Removing noise by adding noise. arXiv:1706.03825.",
        "Street, W. N., Wolberg, W. H. and Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. Proceedings of SPIE 1905, 861–870.",
        "Sundararajan, M., Taly, A. and Yan, Q. (2017). Axiomatic attribution for deep networks. Proceedings of ICML 2017.",
        "Ustun, B., Spangher, A. and Liu, Y. (2019). Actionable recourse in linear classification. Proceedings of FAT* 2019.",
        "Wachter, S., Mittelstadt, B. and Russell, C. (2017). Counterfactual explanations without opening the black box: Automated decisions and the GDPR. Harvard Journal of Law & Technology, 31(2).",
        "Yang, J. et al. (2023). MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data, 10, 41.",
        "Zech, J. R. et al. (2018). Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study. PLOS Medicine, 15(11), e1002683.",
        "Zeiler, M. D. and Fergus, R. (2014). Visualizing and understanding convolutional networks. Proceedings of ECCV 2014.",
    ])


def appendices(w):
    w.sub("Appendix A — Project structure and how to run")
    w.code([
        "medixplain/",
        "├── app.py               Streamlit GUI (6 pages)",
        "├── train_all.py         trains models, runs Trust Lab tests, saves figures + metrics",
        "├── report_assets.py     axiom checks, diagrams and sample cards for this report",
        "├── capture_gui.py       GUI screenshots through headless Edge",
        "├── xai/  tabular.py · image.py · text.py · trust.py · narrative.py · common.py",
        "├── data/  models/  results/",
        "",
        "pip install -r requirements.txt",
        "python train_all.py      # about 2 minutes on a GPU",
        "streamlit run app.py     # opens http://localhost:8502",
    ])
    w.para("Demo flow (5 minutes): Overview → Tumour patient #0 (SHAP waterfall, change the LIME seed, counterfactual, "
           "what-if slider) → Chest X-ray (six maps, then the sanity check with SmoothGrad) → Symptoms (type a sentence, "
           "contrast with the runner-up class) → Trust Lab takeaways.")
    w.sub("Appendix B — Grad-CAM implementation (xai/image.py)")
    w.code([
        "def grad_cam(model, img, target):",
        "    store = {}",
        "    h1 = model.cam_layer.register_forward_hook(lambda m, i, o: store.__setitem__('act', o))",
        "    h2 = model.cam_layer.register_full_backward_hook(",
        "        lambda m, gi, go: store.__setitem__('grad', go[0]))",
        "    try:",
        "        x = _tensor(img).requires_grad_(True)",
        "        model.zero_grad()",
        "        model(x)[0, target].backward()",
        "    finally:",
        "        h1.remove(); h2.remove()",
        "    alpha = store['grad'].mean(dim=(2, 3), keepdim=True)      # α_k: mean gradient per map",
        "    cam = F.relu((alpha * store['act']).sum(1, keepdim=True))  # ReLU(Σ α_k A^k)",
        "    cam = F.interpolate(cam, size=img.shape, mode='bilinear')[0, 0]",
        "    return cam / (cam.max() + 1e-8)",
    ])
    w.sub("Appendix C — Counterfactual search (pseudo-code)")
    w.code([
        "target ← opposite of the predicted class",
        "goal   ← median of the target class in the training set",
        "repeat up to 80 times:",
        "    if the predicted class equals target: stop",
        "    for each feature f and step s in {0.25, 0.5, 1} SD:",
        "        candidate ← move f by s toward goal[f], clipped to the training range",
        "        gain ← change in P(target) / distance moved",
        "    if the best gain > 0: apply that move",
        "    else: apply the largest move on the feature with the highest |SHAP|   # cross a plateau",
        "return every changed feature with its old and new value",
    ])
    w.sub("Appendix D — Sample plain-English explanation cards")
    for label, key in [("Tumour, test patient #0", "tabular"), ("Chest X-ray, test image #0", "image"),
                       ("Symptoms, test description #3", "text")]:
        card = X["cards"][key].replace("_", "")
        w.para(f"**{label}.** " + " ".join(s.strip() for s in card.split("\n\n")))


SECTIONS = [abstract, chapter1, chapter2, chapter3, chapter4, chapter5, chapter6, conclusion, references, appendices]
CHAPTER_NO = {chapter1: 1, chapter2: 2, chapter3: 3, chapter4: 4, chapter5: 5, chapter6: 6}
PLACEHOLDERS = ("<<write the content here>>", "<<list the reference here with bullet>>")


def main(template):
    doc = docx.Document(template)
    add_bullet_numbering(doc)

    # ---- cover and title pages
    # the cover's course code is split across runs as "2" + "3DC2034"
    replace_text_everywhere(doc, "3DC2034", "3DC2022")
    replace_text_everywhere(doc, "OCTOMBER", "OCTOBER")
    _, p = find(doc, "<<include the project title here>>")
    set_paragraph_text(p, TITLE)
    # the title wraps to two lines, so drop one spacer after the course code to keep the cover on one page
    code_p = next(q for q in doc.paragraphs if q.text.startswith("COURSE CODE"))
    spacer = Paragraph(code_p._p.getnext(), code_p._parent)
    if is_empty(spacer):
        remove(spacer)
    _, p = find(doc, "STUDENT NAME (REG.NO)")
    p.runs[0].font.highlight_color = WD_COLOR_INDEX.YELLOW
    _, p = find(doc, "<<The certificate & the mark sheet to be attached here>>")
    for r in p.runs:
        r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    _, p = find(doc, "TITLE")
    set_paragraph_text(p, TITLE)
    p.paragraph_format.line_spacing = 1.5
    _, p = find(doc, "<Font Size 18> <1.5 line spacing>")
    remove(p)
    i, p = next((k, q) for k, q in enumerate(doc.paragraphs) if q.text.startswith("TEAM MEMBERS NAME"))
    set_paragraph_text(p, "STUDENT NAME (REG.NO)", highlight=True)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ---- body: page breaks become "page break before" on each heading
    abstract_idx, _ = find(doc, "ABSTRACT")
    for q in list(doc.paragraphs)[abstract_idx + 1:]:
        if is_pagebreak_only(q):
            remove(q)
    for heading in [f"CHAPTER {k}" for k in range(1, 7)] + ["CONCLUSION", "REFERENCES", "APPENDICES", "EVALUATION SHEET"]:
        _, q = find(doc, heading)
        q.paragraph_format.page_break_before = True

    # ---- fill each placeholder in order
    anchors = [q for q in doc.paragraphs if q.text.strip() in PLACEHOLDERS]
    assert len(anchors) == len(SECTIONS), f"expected {len(SECTIONS)} placeholders, found {len(anchors)}"
    for anchor, fill in zip(anchors, SECTIONS):
        fill(Writer(doc, anchor, CHAPTER_NO.get(fill, 0)))
        # drop the placeholder, its two font hints and the blank spacer lines after it
        nxt = anchor._p.getnext()
        remove(anchor)
        while nxt is not None and nxt.tag == qn("w:p"):
            q = Paragraph(nxt, anchor._parent)
            if q.text.strip().startswith("<<") or is_empty(q):
                after = nxt.getnext()
                nxt.getparent().remove(nxt)
                nxt = after
            else:
                break

    doc.core_properties.title = TITLE
    doc.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Jeskookies\Downloads\XAI IA3 Report Template (1).docx")
