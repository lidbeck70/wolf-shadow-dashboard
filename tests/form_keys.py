"""Formulärrutornas nycklar i testerna. Rutorna heter cf_<TICKER>_<fält>_<avtryck>_<del> (och arkets
ds_<TICKER>_<fält>_<avtryck>), där avtrycket byts när det sparade värdet byts (confidence.ui.stored_tag) — så
testerna slår upp rutan per fält i stället för på exakt nyckel."""
import re

_TAIL = re.compile(r"_(?:0|[0-9a-f]{8})(?:_(v|k|s|t|d))?$")


def field_of(key: str, prefix: str) -> str:
    """Fältnamnet ur en nyckel: cf_GPR_aisc_1a2b3c4d_v → aisc (prefix 'cf_GPR_')."""
    return _TAIL.sub("", str(key)[len(prefix):])


def fields(widgets, prefix: str, part: str = "v") -> set:
    return {field_of(w.key, prefix) for w in widgets
            if str(w.key).startswith(prefix) and _TAIL.search(str(w.key)) and str(w.key).endswith(f"_{part}")}


def box(widgets, prefix: str, field: str, part: str = "v"):
    """Rutan för fältet (text_input/selectbox-listan ur AppTest)."""
    hits = [w for w in widgets if str(w.key).startswith(prefix) and str(w.key).endswith(f"_{part}")
            and field_of(w.key, prefix) == field]
    assert len(hits) == 1, (field, [w.key for w in widgets if str(w.key).startswith(prefix)])
    return hits[0]


def sheet_box(widgets, ticker: str, field: str):
    """Durrett-arkets ruta ds_<TICKER>_<fält>_<avtryck>."""
    prefix = f"ds_{ticker}_"
    hits = [w for w in widgets if str(w.key).startswith(prefix) and _TAIL.search(str(w.key))
            and field_of(w.key, prefix) == field]
    assert len(hits) == 1, (field, [w.key for w in widgets if str(w.key).startswith(prefix)])
    return hits[0]
