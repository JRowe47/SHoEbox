# Retained facts and live composition

Three frozen-component integration runs connect a retained producer to the existing seven-node recurrent graph through one learned latent edge. See the [report](../../../../docs/SHoEbox_Retained_Composition_Report.md) and [locked protocol](../../../../docs/SHoEbox_Retained_Composition_Protocol.md).

The JSON files preserve every prediction, target, table choice, exposure hash and selection record. The artifact attachment contains parent/checkpoint tensors and exact inference exports. Extract it beside the source ZIP, then run `python scripts/audit_retained_composition.py --evaluations` and `python scripts/check_all.py`.
