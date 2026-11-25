# MediXplain: Explainable Clinical AI Workbench

IA3 project for **Explainable AI (23DC2022)**. A Streamlit GUI explains three clinical models, one each for **tabular, image and text** data, with 15 XAI techniques. A **Trust Lab** then tests whether the explanations can be trusted.

| Case | Data | Model | XAI methods |
|---|---|---|---|
| Tumour malignancy | Breast Cancer Wisconsin (tabular) | XGBoost + glass-box LR | SHAP (waterfall, beeswarm, dependence), LIME, permutation importance, PDP/ICE, counterfactuals, what-if |
| Pneumonia | PneumoniaMNIST 64×64 (image) | CNN (PyTorch) | Grad-CAM, Integrated Gradients, SmoothGrad, vanilla saliency, occlusion, LIME superpixels |
| Symptom → diagnosis | symptom_to_diagnosis (text) | TF-IDF + LR | LIME text, SHAP, word occlusion, linear contributions, contrastive "why A not B" |

**Trust Lab:** deletion curves, ERASER comprehensiveness, the Adebayo weight-randomisation sanity check, seed and noise stability, and cross-method agreement.

## Run

```bash
pip install -r requirements.txt
python train_all.py          # trains models, runs evaluations, writes results/ (about 2 minutes on a GPU)
streamlit run app.py         # opens http://localhost:8502
```

To rebuild the Word report from the Karunya template, keep the app running and then:

```bash
python report_assets.py      # axiom checks, architecture diagrams, sample cards
python capture_gui.py        # GUI screenshots (headless Edge, app must be running)
python build_report_docx.py "C:\Users\Jeskookies\Downloads\XAI IA3 Report Template (1).docx"
```

The data is already in `data/`. If it is missing, `train_all.py` cannot fetch the text set, so re-download it:
- PneumoniaMNIST: `python -c "from medmnist import PneumoniaMNIST as P; P(split='train', download=True, size=64, root='data')"`
- Symptoms: `train.jsonl` and `test.jsonl` from huggingface.co/datasets/gretelai/symptom_to_diagnosis, saved as `data/symptom_train.jsonl` and `data/symptom_test.jsonl`.

## Rubric map

| Rubric criterion | Where it is shown |
|---|---|
| Concept understanding | *Concepts & methods* page, REPORT §2 (taxonomy, formulas) |
| Applying to data | Tabular, image and text cases, each with several suitable methods |
| Visualization & interpretation | 20+ interactive plots, highlighted text, plain-English cards, REPORT §5 |
| Presentation | Overview page tells the story. The demo script is below |
| Innovation | Trust Lab, counterfactual search, glass-box comparison, contrastive text, shortcut discovery |
| Documentation | `REPORT.md` (methods, results, discussion, limitations, references) |

## 5-minute demo script

1. **Overview**: three models, one question: "why did it say that, and should we believe the answer?"
2. **Tumour**: on patient #0, show the SHAP waterfall, then LIME. Change the LIME seed to show it moves. Open the Counterfactual tab, then drag a what-if slider.
3. **X-ray**: show all six maps. Switch to the Sanity check tab and pick SmoothGrad, which still looks similar with random weights.
4. **Symptoms**: type your own sentence and watch the words light up. Then show the contrast "why pneumonia not GERD?"
5. **Trust Lab**: show the faithfulness charts (every method beats random), the sanity-check bar chart and the takeaways.
