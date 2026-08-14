"""
Code-Switched and Mixed-Script Multilingual Benchmark Dataset.
Implements Section 3 of Mandatory Specifications:
Contains realistic code-switched multi-script sentence pairs:
- English + Hindi
- English + Spanish
- English + Arabic
- English + Japanese
- English + Korean
- English + Portuguese
- Hindi + English (Devanagari + Latin)
- Tamil + English (Tamil + Latin)
- Telugu + English (Telugu + Latin)
- Arabic + French (Franco-Arabe)
"""

from typing import List, Dict


class CodeSwitchDatasetGenerator:
    """Provides high-utility code-switched evaluation sets across mixed scripts and language pairs."""

    _PAIRS = {
        "cs-en-hi": [
            "I will deploy the tokenizer tomorrow, aur results bilkul accurate aane chahiye.",
            "Please check the configuration file, usme batch size update karna hai.",
            "The model throughput is great, lekin latency thoda optimize karna padega.",
            "Let us run the benchmark suite, sabhi languages test ho jayengi."
        ],
        "cs-en-es": [
            "Let us commit the code changes y luego deploy directly to production.",
            "The database query is very fast, pero necesitamos agregar un nuevo índice.",
            "We have completed the refactoring, todo funciona perfectamente ahora.",
            "Please review the PR and dime si necesitas alguna modificación."
        ],
        "cs-en-ar": [
            "Please review the pull request, إن شاء الله everything is clean and ready.",
            "The training loss converged rapidly, الحمد لله the metrics look promising.",
            "We should update the API endpoints, شكراً جزيلاً for your help.",
            "Let's deploy the new microservices, والله الموفق لكل خير."
        ],
        "cs-en-ja": [
            "The new multilingual tokenizer is working, 本当に素晴らしいパフォーマンスです。",
            "We tested the inference pipeline on GPU, 処理速度が非常に高速になりました。",
            "Please review the pull request, よろしくお願いいたします。",
            "Data preprocessing completed without errors, 次の実験を開始します。"
        ],
        "cs-en-ko": [
            "Let us review the benchmark metrics together, 대단히 감사합니다.",
            "The training job finished successfully, 모델 성능이 크게 향상되었습니다.",
            "We need to optimize the memory footprint, 다음 단계를 진행하겠습니다.",
            "Tokenizer vocabulary is properly balanced, 모든 테스트를 통과했습니다."
        ],
        "cs-en-pt": [
            "We need to fix the cluster node e rodar os testes de integração.",
            "The memory usage is stable, mas precisamos monitorar a latência.",
            "All unit tests passed with success, podemos fazer o deploy agora.",
            "Please check the error logs e me avise se encontrar algum problema."
        ],
        "cs-hi-en": [
            "कृपया server restart करें और production logs check करें।",
            "इस model का accuracy benchmark बहुत high आया है।",
            "हमने tokenizer vocabulary को optimize कर दिया है।",
            "नया pull request review करके merge कर दीजिए।"
        ],
        "cs-ta-en": [
            "இந்த புது tokenizer pipeline-ஐ run பண்ணி test பண்ணுங்க.",
            "நம்ம database cluster ரொம்ப fast-ஆ work ஆகுது.",
            "நாளைக்கு meeting-ல architecture explain பண்ணலாம்.",
            "எல்லா unit tests-உம் pass ஆயிடுச்சு, deploy பண்ணிடலாம்."
        ],
        "cs-te-en": [
            "దయచేసి ఈ pull request review చేసి production లో merge చేయండి.",
            "మా പുതിയ tokenizer model చాలా high compression ratio ఇస్తోంది.",
            "ఈ రోజు benchmark tests అన్నీ complete అయ్యాయి.",
            "కొత్త API endpoints verify చేసి release notes update చేయండి."
        ],
        "cs-ar-fr": [
            "J'ai terminé le rapport technique, كل شيء جاهز pour la réunion de demain.",
            "Les résultats sont excellents, شكراً جزيلاً à toute l'équipe pour les efforts.",
            "On va lancer le nouveau serveur, إن شاء الله sans aucun problème.",
            "Veuillez valider le déploiement, والله ولي التوفيق."
        ]
    }

    @classmethod
    def get_all_pairs(cls) -> Dict[str, List[str]]:
        return cls._PAIRS

    @classmethod
    def get_combined_corpus(cls) -> List[Dict[str, str]]:
        dataset = []
        for pair_code, sentences in cls._PAIRS.items():
            for s in sentences:
                dataset.append({
                    "pair_code": pair_code,
                    "text": s
                })
        return dataset
