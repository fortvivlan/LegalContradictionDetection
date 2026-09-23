import pickle

import pandas as pd

from LCD.shared.bert import _BertBatchCollator, _BertPairDataset


class _PickleableTokenizer:
    pass


def test_bert_dataloader_components_are_pickle_safe_for_spawn_workers() -> None:
    dataframe = pd.DataFrame(
        {
            "premise": ["premise"],
            "hypothesis": ["hypothesis"],
            "tag": ["contradiction"],
        }
    )
    dataset = _BertPairDataset(dataframe, {"contradiction": 0})
    collator = _BertBatchCollator(_PickleableTokenizer(), max_length=512)

    restored_dataset = pickle.loads(pickle.dumps(dataset))
    restored_collator = pickle.loads(pickle.dumps(collator))

    assert restored_dataset[0] == ("premise", "hypothesis", 0)
    assert restored_collator.max_length == 512
    assert "<locals>" not in type(restored_dataset).__qualname__
    assert "<locals>" not in type(restored_collator).__qualname__
