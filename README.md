# Knowing When to Ask a Human: Banking Intent Classification with Calibrated Handoff

ES 335 · Machine Learning 2026 course project. Hypotheses, split rules, and decision criteria are pre-registered in [PREREGISTRATION.md](PREREGISTRATION.md).

- **Live site:** _TBD (GitHub Pages, served from `/docs`)_
- **Data:** BANKING77 (Casanueva et al., 2020, arXiv:2003.04807), CC BY 4.0, downloaded from
  [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets) at a pinned commit.

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m src.download_data
.venv/Scripts/python -m src.split
.venv/Scripts/python -m src.tune
.venv/Scripts/python -m src.evaluate_final   # run once, on frozen models
.venv/Scripts/python -m src.export_web
.venv/Scripts/python -m pytest
```

## AI-use disclosure

Claude (Anthropic) was used for brainstorming, planning, and coding assistance. All results were checked by the author.

## Citations

Casanueva, I., Temčinas, T., Gerz, D., Henderson, M., & Vulić, I. (2020). *Efficient Intent Detection with Dual Sentence Encoders.* Proceedings of the 2nd Workshop on NLP for ConvAI, ACL.
