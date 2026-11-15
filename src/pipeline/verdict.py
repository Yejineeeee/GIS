# -*- coding: utf-8 -*-
"""3단계 판정. 정식 구현(WBS 3.6)은 src/geo/judge.py 에 있다.

5.2 파이프라인과 tools/verdict_compare.py 가 이 파일을 불러 쓰므로, 이름을 그대로 두고
judge.py 를 가져다 쓴다. 판정 규칙은 judge.py 와 docs/verdict.md 를 본다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "geo"))

from judge import (judge, judge_detail, threshold_for, RULE, TARGET,  # noqa: E402,F401
                   THRESHOLDS, THRESHOLDS_95, AMBIGUOUS_SUM, MIN_COVERAGE)
