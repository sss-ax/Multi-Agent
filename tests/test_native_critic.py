from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_runtime.native_critic import parse_critic_verdict


def test_critic_verdict_requires_structured_envelope_and_validates_fields():
    verdict = parse_critic_verdict(
        'The candidate is wrong.\n'
        '<critic_verdict>{"verdict":"needs_repair","confidence":0.92,'
        '"target":"candidate","issues":["bad arithmetic"],'
        '"repair_instructions":"Recalculate the total."}</critic_verdict>'
    )
    assert verdict.valid is True
    assert verdict.verdict == "needs_repair"
    assert verdict.confidence == 0.92
    assert verdict.target == "candidate"
    assert verdict.issues == ("bad arithmetic",)


def test_critic_natural_language_keywords_do_not_control_routing():
    verdict = parse_critic_verdict("The candidate is wrong and needs a fix.")
    assert verdict.valid is False
    assert verdict.verdict == "uncertain"


def test_critic_repair_requires_actionable_instructions():
    verdict = parse_critic_verdict(
        '<critic_verdict>{"verdict":"reject","confidence":1.0,'
        '"target":"candidate","issues":[],"repair_instructions":""}</critic_verdict>'
    )
    assert verdict.valid is False
