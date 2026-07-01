# AdverseEventDetection

Benchmarks LLM-based adverse drug event (ADE) extraction on CADEC-style patient forum posts. Designed to be swapped to n2c2 2018 when portal access is restored.


## Dataset

CADEC contains patient-reported adverse drug event text from health forums. The Hugging Face version used here contains text, ADE mention, and normalized preferred-term fields.

The data is downloaded from KevinSpaghetti/cadec

DatasetDict({
    train: Dataset({
        features: ['text', 'ade', 'term_PT'],
        num_rows: 4745
    })
    test: Dataset({
        features: ['text', 'ade', 'term_PT'],
        num_rows: 1121
    })
})

## Initial Task

Given a patient-reported medication experience, extract all adverse drug events as structured JSON with mention, normalized term, and evidence.
