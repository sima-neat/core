# Strict model archive seeds

`yolo_v9c_seg_mpk.json` is the isolated deterministic MPK seed for model-archive contract tests.
It was extracted from `yolo_v9c_seg_mpk.tar.gz` with parent SHA-256
`dbcf660e59f47acc28a5f7f7b9ec346929592933f866af65be13bac7ed37b2f1`.

The embedded JSON is 47,224 bytes with SHA-256
`bf3c96dd5863446349be7f675c078cfebd8032b658a4ee3355393c5590575f74`.
Its exact terminal topology is 10 ordered heads: three bbox, three class-probability,
three mask-coefficient, and one mask-prototype head. Tests validate this identity and topology
before using the seed; model-zoo caches and environment search paths are intentionally ignored.

`efficientsam3_mpk.json` is the unmodified `mpk.json` of a two-output EfficientSAM3 image model
compiled by Model Compiler 3.0. It is 9,734 bytes with SHA-256
`8fef526843a31fea965b309a13b47060052295e7d336821f0c025a552ea6baad`. Its MLA stage writes the
detections output to the bare `data.ofm.b0` section and the masks output to a per-tensor section;
`unit_mla_elf_io_topology_test` pairs it with a synthesized ELF carrying those section names.
